# EdgeTX test script

`USBTEST.lua` is a small EdgeTX Lua script for the RadioMaster Pocket. It reads a line from the transmitter's USB
serial port, shows it on the screen and writes `ACK:<line>` back. Its file date is 21 November 2026, and it relates
to the idea of relaying PC commands through the transmitter ([feasibility note](../../docs/elrs-relay-feasibility.md)).

It does not send anything to the robot and does not touch the mixer or any channel.

**Status: unverified.** The repository holds no record of this script being run on the transmitter. The feasibility
note found that a Lua script reading USB serial is possible in principle and unproven.
