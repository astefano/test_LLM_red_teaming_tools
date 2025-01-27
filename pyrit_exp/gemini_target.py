# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from httpx import HTTPStatusError
from typing import Optional

from pyrit.common import default_values, net_utility
from pyrit.exceptions import EmptyResponseException, handle_bad_request_exception, pyrit_target_retry
from pyrit.models import PromptRequestResponse
from pyrit.models import construct_response_from_request
from pyrit.prompt_target import PromptTarget, limit_requests_per_minute

logger = logging.getLogger(__name__)


class GeminiTarget(PromptTarget):
    API_KEY_ENVIRONMENT_VARIABLE: str = "GOOGLE_API_KEY"

    def __init__(
        self,
        *,
        endpoint: str,
    ) -> None:
        super().__init__()

        self._endpoint = endpoint
        
    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]

        logger.info(f"Sending the following prompt to the prompt target: {request}")

        try:
            response = await self._complete_text_async(request.converted_value)
            response_entry = construct_response_from_request(request=request, response_text_pieces=[response])
        except HTTPStatusError as bre:
            if bre.response.status_code == 400:
                response_entry = handle_bad_request_exception(
                    response_text=bre.response.text, request=request, is_content_filter=True
                )
            else:
                raise

        return response_entry

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_value_data_type != "text":
            raise ValueError("This target only supports text prompt input.")

    @pyrit_target_retry
    async def _complete_text_async(self, text: str) -> str:
        payload: dict[str, object] = {
            "contents": [{
            "parts": [{"text": text}]
    }]
        }

        resp = await net_utility.make_request_and_raise_if_error_async(
            endpoint_uri=f"{self._endpoint.rstrip('/')}",
            method="POST",
            request_body=payload,
            headers={'Content-Type': 'application/json'},
        )

        if not resp.text:
            raise EmptyResponseException()

        logger.info(f'Received the following response from the prompt target "{resp.text}"')
        return resp.text
