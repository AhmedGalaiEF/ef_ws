• The problem was not your Python code or the robot. It was the network policy on the failing Pi.

  What was happening:

  - Unitree SDK uses DDS/RTPS over UDP, mainly multicast on 239.255.0.1 and ports around 7400-7500.
  - Your failing Pi was receiving those packets on eth0 physically; tcpdump proved that.
  - But the OS was not delivering them cleanly to CycloneDDS because:
      - ufw was enabled with INPUT DROP
      - net.ipv4.conf.eth0.rp_filter was 2

  That caused DDS discovery/matching to fail. When DDS cannot match the remote service endpoint, the SDK times out trying to send the RPC request, which is why you got error 3102
  (RPC_ERR_CLIENT_SEND).

  Why the fix worked:

  - sudo ufw disable
      - stopped dropping inbound UDP multicast/unicast DDS traffic
  - rp_filter=0
      - disabled reverse-path filtering on eth0
      - this matters on a Pi with both wlan0 and eth0, especially with multicast traffic, because Linux can reject packets whose return path or interface check looks inconsistent

  So the chain was:

  1. Robot was sending DDS traffic
  2. Pi could see packets on the wire
  3. Firewall / packet validation blocked proper DDS discovery
  4. SDK could not match video_hub
  5. RPC send timed out with 3102

  After disabling those filters:

  1. DDS discovery succeeded
  2. robot_state calls worked
  3. GetImageSample() matched the robot service
  4. image capture succeeded

  One small note: in ServiceList, the service name is video_hub, not videohub. The capture still worked because the VideoClient uses the correct internal API name.

  If you want, I can give you the minimal permanent config so it keeps working after reboot without fully leaving the machine unprotected.
