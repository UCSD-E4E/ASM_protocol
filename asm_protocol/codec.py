# Description: The codec is intended to decode and encode between bytes and
# lists of packets.
#
# The binaryPacket, E4E_Data_IMU, E4E_Data_Audio_raw8, E4E_Data_Audio_raw16,
# E4E_Data_Raw_File_CTS, E4E_Data_Raw_File_ACK, E4E_Data_Raw_File, and
# binaryPacketParser classes are directly copied from
# https://github.com/UCSD-E4E/e4e-tools/blob/binaryProtocol/binary_protocol/Binary%20Protocol%20Decoder.ipynb

import binascii
import datetime as dt
import enum
import queue
import struct
import uuid
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np


class binaryPacket:
    PACKET_CLASS: int = 0
    PACKET_ID: int = 0

    def __init__(self, payload: bytes, packetClass: int, packetID: int,
                 sourceUUID: uuid.UUID, destUUID: uuid.UUID) -> None:
        self._payload = payload
        self._class = packetClass
        self._id = packetID
        self._source = sourceUUID
        self._dest = destUUID

    def to_bytes(self) -> bytes:
        payloadLen = len(self._payload)
        header = struct.pack("<BB", 0xE4, 0xEB) + \
            self._source.bytes + \
            self._dest.bytes + \
            struct.pack("<BBH", self._class, self._id, payloadLen)
        pktCksum = binascii.crc_hqx(header, 0xFFFF).to_bytes(2, "big")
        msg = header + pktCksum + self._payload
        cksum = binascii.crc_hqx(msg, 0xFFFF).to_bytes(2, "big")
        return msg + cksum

    def getClassIDCode(self) -> int:
        return (self._class << 8) | self._id

    def __str__(self) -> str:
        string = self.to_bytes().hex().upper()
        length = 4
        return ' '.join(string[i:i + length]
                        for i in range(0, len(string), length))

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, packet: Any) -> bool:
        if not isinstance(packet, binaryPacket):
            return False
        return self.to_bytes() == packet.to_bytes()

    @classmethod
    def from_bytes(cls, packet: bytes) -> 'binaryPacket':
        srcUUID, destUUID, pcls, pid, payload = cls.parseHeader(packet)
        src = uuid.UUID(bytes=srcUUID)
        dest = uuid.UUID(bytes=destUUID)
        return binaryPacket(payload, pcls, pid, src, dest)

    @classmethod
    def parseHeader(cls, packet: bytes) -> \
            Tuple[bytes, bytes, int, int, bytes]:
        if binascii.crc_hqx(packet, 0xFFFF) != 0:
            raise RuntimeError("Checksum verification failed")

        if binascii.crc_hqx(packet[0:0x0028], 0xFFFF) != 0:
            raise RuntimeError("Header checksum verification failed")

        if len(packet) < 0x2A:
            raise RuntimeError("Packet too short!")

        s1, s2, pcls, pid, _ = struct.unpack("<BB16x16xBBH", packet[0:0x26])
        srcUUID = bytes(packet[0x0002:0x0012])
        destUUID = bytes(packet[0x0012:0x0022])
        if s1 != 0xE4 or s2 != 0xEB:
            raise RuntimeError("Not a packet!")
        payload = packet[0x28:-2]
        return srcUUID, destUUID, pcls, pid, payload

    @classmethod
    def matches(cls, packetClass: int, packetID: int) -> bool:
        return packetClass == cls.PACKET_CLASS and packetID == cls.PACKET_ID

class E4E_Heartbeat(binaryPacket):
    PACKET_CLASS = 0x01
    PACKET_ID = 0x01
    __VERSION = 0x01
    def __init__(self, src:uuid.UUID, dest:uuid.UUID) -> None:
        super().__init__(b'', self.PACKET_CLASS, self.PACKET_ID, src, dest)

    @classmethod
    def from_bytes(cls, packet: bytes) -> 'E4E_Heartbeat':
        srcUUID, destUUID, _, _, payload = cls.parseHeader(packet)
        if len(payload) != 0:
            raise RuntimeError("Payload not expected")
        src = uuid.UUID(bytes=srcUUID)
        dest = uuid.UUID(bytes=destUUID)
        return cls(src, dest)

class E4E_Data_IMU(binaryPacket):
    PACKET_CLASS = 0x04
    PACKET_ID = 0x00
    __VERSION = 0x01

    def __init__(self, src: uuid.UUID, dest: uuid.UUID, accX: float,
                 accY: float, accZ: float, gyroX: float, gyroY: float,
                 gyroZ: float, magX: float, magY: float, magZ: float,
                 timestamp: dt.datetime = None) -> None:
        self.acc = [accX, accY, accZ]
        self.gyro = [gyroX, gyroY, gyroZ]
        self.mag = [magX, magY, magZ]
        if timestamp is None:
            timestamp = dt.datetime.now()
        self.timestamp = timestamp

        payload = struct.pack("<BBQ3f3f3f",
                              self.__VERSION,
                              0x00,
                              int(timestamp.timestamp() * 1e3),
                              *self.acc,
                              *self.gyro,
                              *self.mag)
        super(E4E_Data_IMU, self).__init__(
            payload, self.PACKET_CLASS, self.PACKET_ID, src, dest)

    @classmethod
    def from_bytes(cls, packet: bytes) -> "E4E_Data_IMU":
        srcUUID, destUUID, _, _, payload = cls.parseHeader(packet)
        ver, _, timestamp_ms, accX, accY, accZ, gyroX, gyroY, gyroZ, magX, \
            magY, magZ = struct.unpack("<BBQ3f3f3f", payload)
        if ver != cls.__VERSION:
            raise RuntimeError("Unknown packet version!")

        timestamp = dt.datetime.fromtimestamp(timestamp_ms / 1e3)
        return E4E_Data_IMU(src=uuid.UUID(bytes=srcUUID),
                            dest=uuid.UUID(bytes=destUUID),
                            accX=accX,
                            accY=accY,
                            accZ=accZ,
                            gyroX=gyroX,
                            gyroY=gyroY,
                            gyroZ=gyroZ,
                            magX=magX,
                            magY=magY,
                            magZ=magZ,
                            timestamp=timestamp)


class E4E_Data_Audio_raw8(binaryPacket):
    PACKET_CLASS = 0x04
    PACKET_ID = 0x01
    __VERSION = 0x01

    def __init__(self, audioData: np.ndarray, src: uuid.UUID, dest: uuid.UUID,
                 timestamp: dt.datetime = None) -> None:
        assert(len(audioData.shape) == 2)
        nChannels = audioData.shape[0]
        nSamples = audioData.shape[1]
        if timestamp is None:
            timestamp = dt.datetime.now()

        self.audioData = audioData
        payload = struct.pack("<BBHQ",
                              self.__VERSION,
                              nChannels,
                              nSamples,
                              int(timestamp.timestamp() * 1e3))
        for channel in range(nChannels):
            for sampleIdx in range(nSamples):
                payload += struct.pack("<B",
                                       int(audioData[channel, sampleIdx]))
        super(E4E_Data_Audio_raw8, self).__init__(
            payload, self.PACKET_CLASS, self.PACKET_ID, src, dest)

    @classmethod
    def from_bytes(cls, packet: bytes) -> 'E4E_Data_Audio_raw8':
        srcUUID, destUUID, pcls, pid, payload = cls.parseHeader(packet)
        ver, nChannels, nSamples, timestamp_ms = struct.unpack(
            "<BBHQ", payload[0:0x0C])
        audioBytes = payload[0x0C:]
        assert(len(audioBytes) == nChannels * nSamples)
        idx = 0
        audioData = np.zeros((nChannels, nSamples))
        for channel in range(nChannels):
            for sampleIdx in range(nSamples):
                audioData[channel, sampleIdx] = audioBytes[idx]
                idx += 1
        timestamp = dt.datetime.fromtimestamp(timestamp_ms / 1e3)
        src = uuid.UUID(bytes=srcUUID)
        dest = uuid.UUID(bytes=destUUID)
        return E4E_Data_Audio_raw8(audioData, src, dest, timestamp)


class E4E_Data_Audio_raw16(binaryPacket):
    PACKET_CLASS = 0x04
    PACKET_ID = 0x02
    __VERSION = 0x01

    def __init__(self, audioData: np.ndarray, src: uuid.UUID, dest: uuid.UUID,
                 timestamp: dt.datetime = None) -> None:
        assert(len(audioData.shape) == 2)
        nChannels = audioData.shape[0]
        nSamples = audioData.shape[1]
        if timestamp is None:
            timestamp = dt.datetime.now()

        self.audioData = audioData
        payload = struct.pack("<BBHQ",
                              self.__VERSION,
                              nChannels,
                              nSamples,
                              int(timestamp.timestamp() * 1e3))
        for channel in range(nChannels):
            for sampleIdx in range(nSamples):
                payload += struct.pack("<H",
                                       int(audioData[channel, sampleIdx]))
        super(E4E_Data_Audio_raw16, self).__init__(
            payload, self.PACKET_CLASS, self.PACKET_ID, src, dest)

    @classmethod
    def from_bytes(cls, packet: bytes) -> 'E4E_Data_Audio_raw16':
        srcUUID, destUUID, pcls, pid, payload = cls.parseHeader(packet)
        ver, nChannels, nSamples, timestamp_ms = struct.unpack(
            "<BBHQ", payload[0:0x0C])
        audioBytes = payload[0x0C:]
        assert(len(audioBytes) == nChannels * nSamples * 2)
        idx = 0
        audioData = np.zeros((nChannels, nSamples))
        for channel in range(nChannels):
            for sampleIdx in range(nSamples):
                audioData[channel, sampleIdx], = struct.unpack(
                    "<H", audioBytes[idx * 2: idx * 2 + 2])
                idx += 1
        timestamp = dt.datetime.fromtimestamp(timestamp_ms / 1e3)
        src = uuid.UUID(bytes=srcUUID)
        dest = uuid.UUID(bytes=destUUID)
        return E4E_Data_Audio_raw16(audioData, src, dest, timestamp)


class E4E_Data_Raw_File_Header(binaryPacket):
    PACKET_CLASS = 0x04
    PACKET_ID = 0xFC
    __VERSION = 0x01

    def __init__(self, fileID: int, filename: str, MIMEType: str,
                 fileSize: int, fileTime: dt.datetime, src: uuid.UUID,
                 dest: uuid.UUID) -> None:
        self.fileID = fileID
        self.filename = filename
        self.mimeType = MIMEType
        self.fileSize = fileSize
        self.fileTime = fileTime
        payload = struct.pack("<BBHHQQ",
                              self.__VERSION,
                              fileID,
                              len(filename),
                              len(MIMEType),
                              fileSize,
                              int(fileTime.timestamp() * 1e3))
        payload += filename.encode('ascii')
        payload += MIMEType.encode('ascii')
        super(E4E_Data_Raw_File_Header, self).__init__(
            payload, self.PACKET_CLASS, self.PACKET_ID, src, dest)

    @classmethod
    def from_bytes(cls, packet: bytes) -> 'E4E_Data_Raw_File_Header':
        srcUUID, destUUID, pcls, pid, payload = cls.parseHeader(packet)
        ver, fileID, filenameLen, mimeTypeLen, fileSize, fileTimestamp = \
            struct.unpack("<BBHHQQ", payload[0:0x16])
        filename = payload[0x16:0x16 + filenameLen].decode()
        mimeType = payload[0x16 + filenameLen:].decode()
        timestamp = dt.datetime.fromtimestamp(fileTimestamp / 1e3)
        src = uuid.UUID(bytes=srcUUID)
        dest = uuid.UUID(bytes=destUUID)
        return E4E_Data_Raw_File_Header(fileID, filename, mimeType, fileSize,
                                        timestamp, src, dest)


class E4E_Data_Raw_File_CTS(binaryPacket):
    PACKET_CLASS = 0x04
    PACKET_ID = 0xFD
    __VERSION = 0x01

    def __init__(self, fileID: int, ack: bool, src: uuid.UUID,
                 dest: uuid.UUID) -> None:
        self.fileID = fileID
        self.ack = ack
        payload = struct.pack("<BBB", self.__VERSION, fileID, int(ack))
        super(E4E_Data_Raw_File_CTS, self).__init__(
            payload, self.PACKET_CLASS, self.PACKET_ID, src, dest)

    @classmethod
    def from_bytes(cls, packet: bytes) -> 'E4E_Data_Raw_File_CTS':
        srcUUID, destUUID, pcls, pid, payload = cls.parseHeader(packet)
        fileID = payload[1]
        if payload[2] == 1:
            ack = True
        else:
            ack = False
        src = uuid.UUID(bytes=srcUUID)
        dest = uuid.UUID(bytes=destUUID)
        return E4E_Data_Raw_File_CTS(fileID, ack, src, dest)


class E4E_Data_Raw_File_ACK(binaryPacket):
    PACKET_CLASS = 0x04
    PACKET_ID = 0xFE
    __VERSION = 0x01

    def __init__(self, fileID: int, seq: int, ack: bool, src: uuid.UUID,
                 dest: uuid.UUID) -> None:
        self.fileID = fileID
        self.seq = seq
        self.ack = ack
        payload = struct.pack("<BBHB", self.__VERSION, fileID, seq, int(ack))
        super(E4E_Data_Raw_File_ACK, self).__init__(
            payload, self.PACKET_CLASS, self.PACKET_ID, src, dest)

    @classmethod
    def from_bytes(cls, packet: bytes) -> 'E4E_Data_Raw_File_ACK':
        srcUUID, destUUID, pcls, pid, payload = cls.parseHeader(packet)
        ver, fileID, seq, ackInt = struct.unpack("<BBHB", payload)
        if ackInt == 1:
            ack = True
        else:
            ack = False
        src = uuid.UUID(bytes=srcUUID)
        dest = uuid.UUID(bytes=destUUID)
        return E4E_Data_Raw_File_ACK(fileID, seq, ack, src, dest)


class E4E_Data_Raw_File(binaryPacket):
    PACKET_CLASS = 0x04
    PACKET_ID = 0xFF
    __VERSION = 0x01

    def __init__(self, fileID: int, seq: int, blob: bytes, src: uuid.UUID,
                 dest: uuid.UUID) -> None:
        self.fileID = fileID
        self.seq = seq
        self.blob = blob
        payload = struct.pack("<BBHQ", self.__VERSION, fileID, seq, len(blob))
        payload += blob
        super(E4E_Data_Raw_File, self).__init__(
            payload, self.PACKET_CLASS, self.PACKET_ID, src, dest)

    @classmethod
    def from_bytes(cls, packet: bytes) -> 'E4E_Data_Raw_File':
        srcUUID, destUUID, pcls, pid, payload = cls.parseHeader(packet)
        ver, fileID, seq, blobLen = struct.unpack("<BBHQ", payload[0:0x0C])
        blob = payload[0x0C:0x0C + blobLen]
        src = uuid.UUID(bytes=srcUUID)
        dest = uuid.UUID(bytes=destUUID)
        return E4E_Data_Raw_File(fileID, seq, blob, src, dest)


class binaryPacketParser:
    class State(enum.Enum):
        FIND_SYNC1 = 0
        FIND_SYNC2 = 1
        HEADER = 2
        HEADER_CKSUM = 3
        HEADER_VALIDATE = 4
        PAYLOAD = 5
        CKSUM = 6
        VALIDATE = 7
        RECYCLE = 8

    packetMap: Dict[int, Type[binaryPacket]] = {
        0x0400: E4E_Data_IMU,
        0x0401: E4E_Data_Audio_raw8,
        0x0402: E4E_Data_Audio_raw16,
        0x04FC: E4E_Data_Raw_File_Header,
        0x04FD: E4E_Data_Raw_File_CTS,
        0x04FE: E4E_Data_Raw_File_ACK,
        0x04FF: E4E_Data_Raw_File
    }

    HEADER_LEN = 0x0026

    def __init__(self) -> None:
        self.__state = self.State.FIND_SYNC1
        self.__payloadLen = 0
        self.__buffer = bytearray()
        self.__data = queue.Queue()

    def parseByte(self, data: int) -> Optional[binaryPacket]:
        self.__data.put(data)
        retval: Optional[binaryPacket] = None
        while not self.__data.empty():
            retval = self._parseByte()
        return retval

    def _parseByte(self) -> Optional[binaryPacket]:
        data = self.__data.get_nowait()

        if self.__state is self.State.FIND_SYNC1:
            if data == 0xE4:
                self.__state = self.State.FIND_SYNC2
                self.__buffer = bytearray()
                self.__buffer.append(data)
            return None
        elif self.__state is self.State.FIND_SYNC2:
            if data == 0xEB:
                self.__state = self.State.HEADER
                self.__buffer.append(data)
            else:
                self.__state = self.State.FIND_SYNC1
            return None
        elif self.__state is self.State.HEADER:
            self.__buffer.append(data)
            if len(self.__buffer) == self.HEADER_LEN:
                self.__state = self.State.HEADER_CKSUM
                self.__payloadLen, = struct.unpack(
                    '<H', self.__buffer[self.HEADER_LEN - 2:self.HEADER_LEN])
            return None
        elif self.__state is self.State.HEADER_CKSUM:
            self.__buffer.append(data)
            self.__state = self.State.HEADER_VALIDATE
            return None
        elif self.__state is self.State.HEADER_VALIDATE:
            self.__buffer.append(data)
            if binascii.crc_hqx(self.__buffer, 0xFFFF) != 0:
                self.__state = self.State.FIND_SYNC1
                self.__recycleBuffer = self.__buffer[2:]
                while not self.__data.empty():
                    self.__recycleBuffer.append(self.__data.get_nowait())
                for byte in self.__recycleBuffer:
                    self.__data.put(byte)
            else:
                self.__state = self.State.PAYLOAD
            return None
        elif self.__state is self.State.PAYLOAD:
            self.__buffer.append(data)
            if len(self.__buffer) == self.__payloadLen + self.HEADER_LEN + 2:
                self.__state = self.State.CKSUM
            return None
        elif self.__state is self.State.CKSUM:
            self.__buffer.append(data)
            self.__state = self.State.VALIDATE
            return None
        elif self.__state is self.State.VALIDATE:
            self.__buffer.append(data)
            if binascii.crc_hqx(self.__buffer, 0xFFFF) != 0:
                raise RuntimeError("Checksum verification failed")
            packetID, = struct.unpack('>H', self.__buffer[0x0022:0x0024])
            self.__state = self.State.FIND_SYNC1
            if packetID not in self.packetMap:
                return binaryPacket.from_bytes(self.__buffer)
            else:
                return self.packetMap[packetID].from_bytes(self.__buffer)
        else:
            return None

    def parseBytes(self, data: bytes) -> List[binaryPacket]:
        packets = []
        for byte in data:
            self.__data.put(byte)
        while not self.__data.empty():
            retval = self._parseByte()
            if retval is not None:
                packets.append(retval)
        return packets


class Codec:
    def decode(self, data: bytes) -> List[binaryPacket]:
        parser = binaryPacketParser()
        binaryPackets = parser.parseBytes(data)
        return binaryPackets

    def encode(self, data: List[binaryPacket]) -> bytes:
        bytesData = []
        for packet in data:
            bytesObj = packet.to_bytes()
            bytesData.append(bytesObj)
        binaryBytes = b"".join(bytesData)
        return binaryBytes
