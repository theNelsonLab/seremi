from pathlib import Path
import numpy as np
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
import mmap

assert sys.byteorder == 'little', 'EMI reader assumes little endian host'


class EMIFile:
    img_arr: np.ndarray
    domain: str
    original_filepath: str
    info_xml: dict

    def __init__(self, path: str | Path):
        # based on https://github.com/zhijie-li/TIA_dump
        with open(path, 'rb') as f:
            buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

            magic_header = buf[0:12]
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
            img_magic_pos = buf.find(IMG_MAGIC, 1000, 1050)

            img_dtype_id = int(buf[img_magic_pos - 1])
            assert img_dtype_id == 6, 'only dtype 6 (int32) is supported'

            img_nums_start = img_magic_pos + len(IMG_MAGIC)
            img_data_start = img_nums_start + 12
            img_header_nums = np.frombuffer(buf[img_nums_start:img_data_start], dtype=np.uint32)
            img_len_bytes = int(img_header_nums[0]) - 8
            img_width, img_height = int(img_header_nums[1]), int(img_header_nums[2])

            assert img_len_bytes == img_width * img_height * 4, 'calculated wrong image size based on dimensions'

            img_data_end = img_data_start + img_len_bytes
            self.img_arr = np.frombuffer(buf[img_data_start:img_data_end], dtype=np.int32).reshape(
                (img_height, img_width))

            footer = buf[img_data_end:]
            self.domain = EMIFile._read_footer_string(footer, b'\x00\x04')
            self.original_filepath = EMIFile._read_footer_string(footer, b'\x40\x04')
            info_xml_str = EMIFile._read_footer_string(footer, b'\x19\x04')
            self.object_info = EMIFile._xml_to_dict(info_xml_str)

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