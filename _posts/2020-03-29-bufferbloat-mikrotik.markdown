---
layout: post
title:  "Bufferbloat and Mikrotik Router"
description: Learning about home networking and preventing Bufferbloat with better QoS in Mikrotik router.
date:   2020-03-29 16:00:00 +0
categories: networking WFH
---
Due to [covid-19][fa-covid] social (distancing) responsibility, a lot of people are now having to work from home (WFH). For me, this means having to connect to the company [VPN][vpn] for file access, remote-desktop for data visualisation on [HPC][hpc], and online-only communications including frequent video calls and screen-sharing.

With a sudden spike in network traffic (from everyone WFH!), company network bandwidth can obviously become a bottleneck. However, besides bandwidth, network [_latency_][latency] (a.k.a. 'lag' – which reminds me of Counter Strike Beta 6.0…) can also be a problem, e.g. for remote-desktop and screen-sharing.

From my recent WFH network activities, and partly by pure chance, I happen to stumble into —or at least to the entrance of…— the rabbit hole of home networking tweaks, including learning about something called [Bufferbloat][bloat] that affects latency, and how to mitigate it.

### Bufferbloat

The [Bufferbloat][bloat] Project explains Bufferbloat as "_the undesirable latency that comes from a router or other network equipment buffering too much data._" Their [wiki][bloatwiki] suggests a simple measure of Bufferbloat using [DSLReports Speed Test][dsl]. When I ran the speed test, I got results that looked like the following:

{% include screenshot url="speedtest_noQoS.png" %}

My broadband package is 50Mbit/s down, 5Mbit/s up, and so it _looks_ like I am getting my money's worth (extra ~10% down-link bandwidth!), but the latency seems affected by this Bufferbloat problem, where I observed occasional latencies of >200ms during the speed test:

{% include screenshot url="speedtest_bloat.png" %}

Their [suggested solution][solution] is to use [Smart Queue Management][sqm] (SQM) in your router, but my —actually, most?— router does not have SQM, unfortunately… However, they did say that [QoS][qos] (which is more widely available in routers) can help, even though it [will not solve][solqos] Bufferbloat completely. And so I can still give it a go~

With my newly setup [Mikrotik][mk] [router][ac2] (more on that [below](#mikrotik-for-home-use)), a bit more Googling brought me to [this][config] nice page, where they have a simple Mikrotik [QoS config][qosconfig] tool. My internet connection details are simple enough, with just a single WAN internet connection to Mikrotik's `ether1` port, and a single `bridge` (for ethernet and WiFi) on the LAN interface side:

{% include screenshot url="mikrotik_interface.png" %}

Up-link and down-link speeds seem to follow my broadband package specs (see speed test above), and so I can just use `5M` and `50M` on the config tool webpage:

{% include screenshot url="qos_config.png" %}

I left the rest of the settings alone, and just downloaded the resulting script. To be safe, I had a quick look at it in a text editor, and then followed the instructions to import the config script into Mikrotik [Winbox][winbox]. I also had to go into IP - Firewall - Filter Rules, and removed the 'FastTrack' rule so that the new QoS config settings will apply, instead of being bypassed by 'FastTrack'.

The generated config seems to use a queue type called [_PCQ_][pcq], even though another [site about lag/latency][stoplag] mentioned the use of a different queue type called [_SFQ_][sfq] for Bufferbloat (in the absence of the preferred [SQM][sqm] method). But, what the hell, they all don't make much sense to me anyways, so I'll just try the config I've got!

Then, the moment of truth. I re-ran the [DLSReports Speed Test][dsl], and now it's saying that the Bufferbloat problem is gone, with the rating improving from B to A+, though it looks like the QoS bandwidth limits have caused the speeds to reduce slightly, as a trade-off:

{% include screenshot url="speedtest_pcq.png" %}

I have since tested a few different speed limit (approx. ±10%) settings in the QoS config, but in the end still settled on the 'rated speeds' for my broadband package. I also tweaked the QoS service list and protocol/port settings, to change the priorities for my own use cases, while also adding new services such as [Microsoft Teams][teams] to higher priority, for video meetings etc.

Anecdotally, it seems like the overall network performance has improved with the configured Mikrotik router, and video calls on various software seem to perform well, with less 'lag' than before. I guess the best solution to prevent Bufferbloat is still to use a router that supports [SQM][sqm] e.g. via [OpenWrt][openwrt] firmware, but the routers can be expensive, and even the [more affordable ones][c7] look like they will have [trade-offs][c7review] in other features in an all-in-one WiFi router. Overall, I am happy to stick with the new Mikrotik, which brings us to how I started using it in the first place…

### Mikrotik for home use

Even before the extra WFH traffic, my old Netgear WiFi router was already starting to act up, with its 5GHz WiFi occasionally dropping for no reason. I did a bit of research, and read that [Mikrotik][mk] gear (frequently used by [SOHO][soho]s) can be a cost-effective, high-performance home router. A bit more Googling led me to the [Mikrotik hAP ac²][ac2], which seems to fit my requirements:
* Router and WiFi access point all-in-one
* [Dual-Concurrent][blades] 2.4/5GHz AP, supporting up to [802.11ac][ac] WiFi
* Five Gigabit ethernet ports for [WAN][wan] and [LAN][lan]-wired devices
* Small unit, with no crazy antennas, but enough coverage for a small place
* Relatively cheap for its features, c.£65 on [Amazon][amazon]

I found one for <£60 from an eBay seller, and went for it. Then came the fun(?) of setting it up for home networking use!

Unlike normal retail WiFi routers, this required a bit more work. For the initial setup, I pretty much just followed [this great guide][ligos] (thanks, Murray!). As I bought my Mikrotik from eBay, before doing anything, I did a full reset of the router. I did not have to do the "_Make You Old Router Into a Modem_" step, because my fibre-optic ISP-supplied router (WiFi functionality already disabled from before) can just be connected directly to the Mikrotik ethernet port 1 as the WAN connection. I did not hear any beeps (mentioned in [step 2][ligos]) when I powered up the Mikrotik, but I guess that differs from model to model.

Following [step 3][ligos]:
* I used [Winbox][winbox]'s Quick Set to setup the local network (step 3a) and system password (3b).
* For WiFi (3c), I only setup 5GHz WiFi, and disabled the 2.4GHz WiFi because none of the devices at home will need it.
* I also checked the 5GHz WiFi frequency ranges occupied by my neighbours, and picked a freq. range that appeared free.
* I double-checked that [WPS][wps] is completely disabled, because I do not plan to use it, and it can be a bit [dodgy][wps].
* I skipped step 3d ("_Internet_") because my fibre-optic broadband is already connected as WAN.
* Step 3e ("_Updates_") showed that the Mikrotik I bought was already up-to-date in its [firmware and packages][download].
* I will circle back to the setup of Guest WiFi (3f) later.
* And I skipped step 3g because I do not plan to have a VPN server running at home (plus my broadband is without static IP or port-forwarding…)

Then, I had a look at the various things mentioned in [step 4][ligos]. The interfaces all looked okay, and the 5GHz WiFi signal looked good even in the room furthest from the Mikrotik, which is amazing for such a small box (compared to the much bigger old Netgear!). When looking at the DHCP server, I also setup new static IPs for [NAS][nas] and desktop PC (sometimes used via [RDP][rdp]). For DNS servers, I ran the super informative [DNS benchmark tool from GRC][grc], which confirmed that [CloudFlare][dns]'s 1.1.1.1 (primary) and 1.0.0.1 (secondary) servers are my best bet, by far. The [guide][ligos] also recommended turning off unused IP Services to reduce attack surface, and this is where I referred to [further steps][secure] (besides keeping things up-to-date) on securing the Mikrotik router, as there have been [major vulnerabilities][cve] before, though mainly affecting out-of-date firmware.

I did not actually do much more for [steps 5 and 6][ligos], as the default firewall rules looked okay for my use, and I am not using IPv6 for my home network. In the last section just before his Conclusion, Murray's [guide][ligos] mentioned [Queues][queue] and [QoS][qos]. The tip given is to not use _fifo_ queues, but to use _sfq_ or _pcq_ queues to prevent [Bufferbloat][bloat], though without much detail. None of these meant anything to me at all, at the time..! But I looked into it further, and I _think_ I got [something useful](#bufferbloat) out of it.

### Isolated Guest WiFi setup

Next, I circled back to step 3f of [this guide][ligos], to setup an isolated guest WiFi that sits on a different subnet IP range and prevented from accessing the LAN devices on my network. There was only a brief list of basic descriptions in the [guide][ligos], and so I found another more detailed one [here][marthur] (thanks, Marthur!) to follow.

[Marthur][marthur]'s steps are pretty clear, and the setup of Virtual WiFi AP, VLAN, firewall rules, etc. were straightforward enough. The only extra thing I had to do was to create a new '[interface list][list]' to include both my original LAN Bridge and the new Guest WiFi Bridge, and then modify the [Firewall Mangle][mangle] rules (from the [QoS config][config] script; see [above](#bufferbloat)) so that the QoS rules are now applied to all traffic (including to/from the new Guest WiFi). Then I just quickly hopped on to the Guest WiFi, confirmed that it has internet connectivity but no access to LAN devices, and then double-checked on [Speed Test][dsl] that it still honoured the QoS settings and did not cause [Bufferbloat](#bufferbloat). And that's the Guest WiFi done~!

### Mikrotik automatic backup and update

Finally, wary of potential [vulnerabilities][cve] if firmware/packages go out-of-date, I found and followed [this][reddit] (thanks, _/u/beeyev_!) to setup automatic backup and update for the Mikrotik router. A quick look at the [script][beeyev] did not throw up any obvious red flags, so I just imported it into [Winbox][winbox] and set it up according to the clearly commented instructions. I enabled the setting to install only _patch_ minor version updates, and also setup auto-email whenever the script runs (scheduled for every two days). The email feature needs an [SMTP][smtp] server, so I just followed the recommendation and used the excellent free service on [smtp2go][smtp2go]. One small note: in Mikrotik's [Email][email] settings, for "Start TLS" the `tls_only` option did not work for me, so I chose `yes` instead, and it all seems to work fine:

{% include screenshot url="email_tls.png" %}

That's all for this post. Time to get back to WFH with strange working hours…

[fa-covid]: https://www.fast.ai/2020/03/09/coronavirus/
[vpn]: https://en.wikipedia.org/wiki/Virtual_private_network
[hpc]: https://en.wikipedia.org/wiki/High-performance_computing
[latency]: https://en.wikipedia.org/wiki/Latency_(engineering)#Packet-switched_networks
[bloat]: https://www.bufferbloat.net/projects/
[mk]: https://mikrotik.com/
[soho]: https://en.wikipedia.org/wiki/Small_office/home_office
[ac2]: https://mikrotik.com/product/hap_ac2
[wan]: https://en.wikipedia.org/wiki/Wide_area_network
[lan]: https://en.wikipedia.org/wiki/Local_area_network
[ac]: https://en.wikipedia.org/wiki/IEEE_802.11ac
[amazon]: https://www.amazon.co.uk/MikroTik-RBD52G-5HACD2HND-TC-hAP-ac2/dp/B079SD8NVQ
[ligos]: https://blog.ligos.net/2017-02-16/Use-A-Mikrotik-As-Your-Home-Router.html
[winbox]: https://mt.lv/winbox64
[wps]: https://en.wikipedia.org/wiki/Wi-Fi_Protected_Setup#Vulnerabilities
[download]: https://mikrotik.com/download
[nas]: https://en.wikipedia.org/wiki/Network-attached_storage
[rdp]: https://en.wikipedia.org/wiki/Remote_Desktop_Protocol
[grc]: https://www.grc.com/dns/benchmark.htm
[dns]: https://1.1.1.1/
[secure]: https://wiki.mikrotik.com/wiki/Manual:Securing_Your_Router
[cve]: https://www.cvedetails.com/vulnerability-list/vendor_id-12508/product_id-23641/Mikrotik-Routeros.html
[queue]: https://wiki.mikrotik.com/wiki/Manual:Queue
[qos]: https://en.wikipedia.org/wiki/Quality_of_service
[bloatwiki]: https://www.bufferbloat.net/projects/bloat/wiki/
[dsl]: http://www.dslreports.com/speedtest
[config]: https://mikrotikconfig.com/
[qosconfig]: https://mikrotikconfig.com/qos/
[teams]: https://docs.microsoft.com/en-us/microsoftteams/upgrade-prepare-environment-prepare-network
[marthur]: https://www.marthur.com/networking/mikrotik-setup-guest-vlan-wifi/2582/
[solution]: https://www.bufferbloat.net/projects/bloat/wiki/What_can_I_do_about_Bufferbloat/
[sqm]: https://www.bufferbloat.net/projects/cerowrt/wiki/Smart_Queue_Management/
[solqos]: https://www.bufferbloat.net/projects/bloat/wiki/More_about_Bufferbloat/#what-s-wrong-with-simply-configuring-qos
[stoplag]: https://www.stoplagging.com/mikrotik-box-method/
[pcq]: https://wiki.mikrotik.com/wiki/Manual:Queues_-_PCQ
[sfq]: https://wiki.mikrotik.com/wiki/Manual:Queue#SFQ
[c7]: https://openwrt.org/toh/tp-link/archer-c7-1750
[c7review]: https://www.techradar.com/uk/reviews/pc-mac/networking-and-wi-fi/modem-routers/tp-link-archer-c7-ac1750-1198451/review
[openwrt]: https://openwrt.org/
[blades]: https://www.youtube.com/watch?v=_eRRab36XLI
[list]: https://wiki.mikrotik.com/wiki/Manual:Interface/List
[mangle]: https://wiki.mikrotik.com/wiki/Manual:IP/Firewall/Mangle
[reddit]: https://www.reddit.com/r/mikrotik/comments/ercpzb/mikrotik_routeros_automatic_backup_and_update/
[beeyev]: https://github.com/beeyev/Mikrotik-RouterOS-automatic-backup-and-update
[smtp2go]: https://smtp2go.com/
[smtp]: https://en.wikipedia.org/wiki/Simple_Mail_Transfer_Protocol
[email]: https://wiki.mikrotik.com/wiki/Manual:Tools/email
