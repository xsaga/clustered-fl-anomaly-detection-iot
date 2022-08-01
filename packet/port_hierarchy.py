
port_basic_three_map = [
    (range(0, 1024), "system"),
    (range(1024, 49152), "user"),
    (range(49152, 65536), "dynamic")
]

port_hierarchy_map = [
    ([80,280,443,591,593,777,488,1183,1184,2069,2301,2381,8008,8080], "httpPorts"),
    ([24,25,50,58,61,109,110,143,158,174,209,220,406,512,585,993,995], "mailPorts"),
    ([42,53,81,101,105,261], "dnsPorts"),
    ([20,21,47,69,115,152,189,349,574,662,989,990], "ftpPorts"),
    ([22,23,59,87,89,107,211,221,222,513,614,759,992], "shellPorts"),
    ([512,514], "remoteExecPorts"),
    ([13,56,113,316,353,370,749,750], "authPorts"),
    ([229,464,586,774], "passwordPorts"),
    ([114,119,532,563], "newsPorts"),
    ([194,258,531,994], "chatPorts"),
    ([35,92,170,515,631], "printPorts"),
    ([13,37,52,123,519,525], "timePorts"),
    ([65,66,118,150,156,217], "dbmsPorts"),
    ([546,547,647,847], "dhcpPorts"),
    ([43,63], "whoisPorts"),
    (range(137,139+1), "netbiosPorts"),
    ([88,748,750], "kerberosPorts"),
    ([111,121,369,530,567,593,602], "RPCPorts"),
    ([161,162,391], "snmpPorts"),
    (range(0,1024), "PRIVILEGED_PORTS"),
    (range(1024,65536), "NONPRIVILEGED_PORTS")
]

def port_to_categories(port_map, port: int) -> str:
    for ph_range, ph_name in port_map:
        if port in ph_range:
            return ph_name

    return ""