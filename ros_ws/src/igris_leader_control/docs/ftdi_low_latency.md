# IGRIS leader FTDI low-latency setup

This package contains an opt-in udev rule for the known leader adapter only:

- USB vendor/product: `0403:6014`
- USB serial: `FTAU59R6`
- requested FTDI latency timer: `1 ms`

The template does not change the running system merely by building or installing
the ROS package. It is intentionally scoped to the adapter serial number so that
other FTDI devices are unaffected.

## Review and install

Do this only while the robot is safe, the leader process is stopped, and the
leader adapter can be disconnected. From the package source directory:

```bash
udevadm test-builtin usb_id /sys/class/tty/ttyUSB0 2>/dev/null | \
  grep -E 'ID_VENDOR_ID=0403|ID_MODEL_ID=6014|ID_SERIAL_SHORT=FTAU59R6'
sudo install -m 0644 udev/99-igris-leader-ftdi-low-latency.rules \
  /etc/udev/rules.d/99-igris-leader-ftdi-low-latency.rules
sudo udevadm control --reload-rules
```

With the leader node stopped, either disconnect/reconnect the adapter or apply
the rule to its existing `usb-serial` device and wait for udev:

```bash
sudo udevadm trigger --action=add /sys/bus/usb-serial/devices/ttyUSB0
sudo udevadm settle
```

Do not trigger it while a leader node has the serial port open. Then verify the
adapter identity and timer without writing to sysfs:

```bash
udevadm info --query=property --name=/dev/ttyUSB0 | \
  grep -E 'ID_VENDOR_ID=|ID_MODEL_ID=|ID_SERIAL_SHORT='
cat /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
```

The expected timer output is `1`. If the device enumerates under a different
`ttyUSB` number, use that name only for verification; the rule matches hardware
identity rather than the device number.

## Validate before keeping the rule

Run the leader with its default 100 Hz publish target and leave the timing report
enabled for at least 30 minutes:

```bash
export IGRIS_LEADER_TIMING_INTERVAL_S=10
ros2 run igris_leader_control leader_node --ros-args \
  -p device_name:=/dev/ttyUSB0 -p publish_hz:=100.0
```

Each `leader_timing` line reports windowed p50/p95/p99/max values for callback
period (`loop_ms`), Dynamixel group transaction (`txrx_ms`), and complete callback
execution (`callback_ms`), plus published and dropped sample counts. Compare these
logs before and after enabling the rule. Do not raise baud rate or control-loop
frequency in the same experiment.

Initial acceptance targets are:

- at least 95 successful publishes per second over a stable window;
- `loop_ms` p99 below 15 ms;
- `txrx_ms` p99 below 6 ms;
- zero dropped samples or Dynamixel communication errors for 30 minutes.

These are commissioning targets, not a substitute for end-to-end robot latency
and stability tests.

## Remove

To restore the system default, stop the leader, remove the installed rule, reload
udev, then disconnect and reconnect the adapter:

```bash
sudo rm /etc/udev/rules.d/99-igris-leader-ftdi-low-latency.rules
sudo udevadm control --reload-rules
```

The project-local template remains available for later review.
