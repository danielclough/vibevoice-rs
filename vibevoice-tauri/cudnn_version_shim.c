// Minimal shim for cudarc on Windows + cuDNN 8.9.x
// These version check functions were removed in newer cuDNN versions,
// but cudarc still references them. Returning 0 indicates success.
// See: https://github.com/coreylowman/cudarc/issues/cudnn-version-check

__declspec(dllexport) int cudnnAdvVersionCheck(void) { return 0; }
__declspec(dllexport) int cudnnCnnVersionCheck(void) { return 0; }
__declspec(dllexport) int cudnnOpsVersionCheck(void) { return 0; }
