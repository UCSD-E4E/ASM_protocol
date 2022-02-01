import datetime as dt
from uuid import uuid1

import asm_protocol.codec as asm_codec
import random

def test_dataLabels():
    codec = asm_codec.Codec()
    random.seed(0)

    for _ in range(65536):
        source_uuid = uuid1()
        dest_uuid = uuid1()
        now = dt.datetime.now()
        label = random.randint(0, 2**64 - 1)

        data_packet = asm_codec.E4E_Data_Labels(label, sourceUUID=source_uuid, destUUID=dest_uuid, timestamp=now)

        blob = codec.encode([data_packet])
        packets = codec.decode(blob)
        assert(len(packets) == 1)
        packet = packets[0]
        assert(isinstance(packet, asm_codec.E4E_Data_Labels))

        assert(packet._source == source_uuid)
        assert(packet._dest == dest_uuid)
        assert(abs((packet.timestamp - now).total_seconds()) < 1e-3)
        assert(packet.label == label)
