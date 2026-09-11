# Robot-matched hand SDK snapshot

This directory is required source-distribution material, despite the parent
directory's name. `ros_ws/src/igris_c_ros_bridge/igris_c_hand/CMakeLists.txt`
explicitly uses these two files:

- `include/igris_sdk/igris_c_msgs.hpp`
- `lib/libigris_sdk.a`

They provide the HandCmd/HandState wire schema used by the robot version with
which this workspace was configured. The public SDK vendored under
`ros_ws/src/igris_c_ros_bridge/thirdparty/igris_c_sdk_public` remains a separate
dependency for the body and other bridges. Substituting its header/library for
this pair is not a compatible migration procedure.

The pair already existed on the original PC; publication preserves it unchanged.
An upstream commit/version is not recorded in the snapshot, so none is asserted
here. `SHA256SUMS` identifies the exact distributed bytes. Retain the upstream
SDK's licensing terms; no new redistribution license is granted by this project.
The static library is architecture/toolchain-specific: the documented baseline
is Ubuntu 24.04 x86_64. Other architectures require a matching SDK build, not a
blind copy of this library.

Verify from this directory:

```bash
sha256sum -c SHA256SUMS
```
