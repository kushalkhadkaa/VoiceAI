from __future__ import annotations

VOICE_LICENSES = {
    "ne_NP-chitwan-medium": {
        "license": "Creative Commons Attribution 4.0 International (CC-BY-4.0)",
        "status": "allowed"
    },
    "ne_NP-google-medium": {
        "license": "Creative Commons Attribution 4.0 International (CC-BY-4.0)",
        "status": "allowed"
    },
    "en_US-lessac-medium": {
        "license": "Creative Commons Attribution 4.0 International (CC-BY-4.0)",
        "status": "allowed"
    },
    "en_US-ryan-medium": {
        "license": "Creative Commons Attribution 4.0 International (CC-BY-4.0)",
        "status": "allowed"
    }
}


class ModelLicenseRegistry:
    @staticmethod
    def get_license_info(voice_id: str) -> dict[str, str]:
        return VOICE_LICENSES.get(
            voice_id,
            {
                "license": "unknown; review model card before commercial use",
                "status": "needs_review"
            }
        )
