from pathlib import Path
import numpy as np
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
import mmap
from typing import BinaryIO


class EMIFile:
    """
    Read an EMI file, which contains detailed metadata and the last frame of an SER.
    """

    #: height of the image in pixels
    img_height: int
    #: width of the image in pixels
    img_width: int
    #: domain metadata field, such as "Reciprocal Space"
    domain: str
    #: the filepath of the SER when it was saved the first time
    original_filepath: str
    #: metadata XML object converted to dictionary
    info_dict: dict

    _path: Path
    _file: BinaryIO
    _buf: mmap.mmap
    _img_data_start: int
    _img_data_end: int

    def __init__(self, path: str | Path):
        self._path = path

    def open(self):
        """
        Open the EMI and populate metadata fields.
        This method does not read frame contents.
        """
        if sys.byteorder != 'little':
            raise OSError('EMI reader expects little endian host')

        self._file = open(self._path, 'rb')
        self._buf = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

        # based on https://github.com/zhijie-li/TIA_dump
        magic_header = self._buf[0:12]
        assert magic_header == b'\x4A\x4B\x00\x02\x00\x00\x00\x00\x04\x4D\x01\x00'

        # the file header is approximately 1000 bytes long, slightly varying
        # depending on the version of the writer software. It seems to be composed
        # of many different fields, of different data types. After this header,
        # there is an image blob, which is what we want to extract.
        # The image blob header follows the format (following zhijie-li):
        # 2B '00 00'		   ending last segment
        # 4B '20 00 00 02'	   annoucing a int8
        # 1B int8                datatype =6 means 4-byte int signed
        # 4B '08 22 02 02'       unknown, likely datablock node tag
        # 12B 3xint4             datasize+8bytes, width, hight
        # <dataarray> of datasize bytes

        IMG_MAGIC = b'\x08\x22\x02\x02'
        img_magic_pos = self._buf.find(IMG_MAGIC, 1000, 1050)

        img_dtype_id = int(self._buf[img_magic_pos - 1])
        assert img_dtype_id == 6, 'only dtype 6 (int32) is supported'

        img_nums_start = img_magic_pos + len(IMG_MAGIC)
        self._img_data_start = img_nums_start + 12
        img_header_nums = np.frombuffer(self._buf[img_nums_start:self._img_data_start], dtype=np.uint32)
        img_len_bytes = int(img_header_nums[0]) - 8
        self.img_width, self.img_height = int(img_header_nums[1]), int(img_header_nums[2])

        assert img_len_bytes == self.img_width * self.img_height * 4, 'calculated wrong image size based on dimensions'

        self._img_data_end = self._img_data_start + img_len_bytes

        footer = self._buf[self._img_data_end:]
        self.domain = EMIFile._read_footer_string(footer, b'\x00\x04')
        self.original_filepath = EMIFile._read_footer_string(footer, b'\x40\x04')
        info_xml_str = EMIFile._read_footer_string(footer, b'\x19\x04')
        self.info_dict = EMIFile._xml_to_dict(info_xml_str)

    def __enter__(self):
        self.open()
        return self

    def close(self):
        """
        Close internal resource handles
        """
        self._buf.close()
        self._file.close()

    def __exit__(self, *_args):
        self.close()

    def read_frame(self) -> np.ndarray:
        """
        Extract the one frame stored inside the EMI file.
        In my testing, this image is always the last frame of the corresponding SER file.
        :return: 2D int32 array
        """
        return (
            np.frombuffer(self._buf[self._img_data_start:self._img_data_end], dtype=np.int32)
            .reshape((self.img_height, self.img_width))
        )

    @staticmethod
    def _read_footer_string(footer: bytes, key: bytes) -> str:
        # Strings in the footer are in the format
        # <60 00> <2 byte key> <length uint32_le> <string bytes>
        search_bytes = b'\x60\x00' + key
        header_start_pos = footer.find(search_bytes)
        len_start_pos = header_start_pos + len(search_bytes)
        chars_start_pos = len_start_pos + 4
        str_len = int(np.frombuffer(footer[len_start_pos:chars_start_pos], dtype=np.uint32)[0])
        assert chars_start_pos + str_len <= len(footer), "string too long"
        return footer[chars_start_pos:chars_start_pos + str_len].decode()

    @staticmethod
    def _xml_to_dict(xml_str):
        return EMIFile._etree_to_dict(ET.fromstring(xml_str))

    @staticmethod
    def _etree_to_dict(t):
        # https://stackoverflow.com/a/10077069
        d = {t.tag: {} if t.attrib else None}
        children = list(t)
        if children:
            dd = defaultdict(list)
            for dc in map(EMIFile._etree_to_dict, children):
                for k, v in dc.items():
                    dd[k].append(v)
            d = {t.tag: {k: v[0] if len(v) == 1 else v for k, v in dd.items()}}
        if t.attrib:
            d[t.tag].update(('@' + k, v) for k, v in t.attrib.items())
        if t.text:
            text = t.text.strip()
            if children or t.attrib:
                if text:
                    d[t.tag]['#text'] = text
            else:
                d[t.tag] = text
        return d
