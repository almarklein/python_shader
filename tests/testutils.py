import os
import subprocess


def _determine_use_vulkan_sdk():
    if os.getenv("PYTHON_SHADER_TEST_FULL", "").lower() == "true":
        return True
    else:
        try:
            subprocess.check_output(["spirv-val", "--version"])
        except Exception:
            return False
        else:
            return True


use_vulkan_sdk = _determine_use_vulkan_sdk()
