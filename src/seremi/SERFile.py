from pathlib import Path
from typing import BinaryIO
import numpy as np
import sys
import struct
import mmap


class SERFile:
    """
    Read a SER microscope image file, which contains multiple frames. Initially, on opening an SER file, only metadata is read.
    To access image content of a particular frame, call read_frame().

    SER files contain very little metadata. Instead, for a SER file "abc_1.ser", there exists a corresponding EMI file "abc.emi",
    which contains detailed metadata corresponding to the SER.
    """

    #: user-provided path of the image file
    path: Path
    #: height of each image in pixels
    img_height: int
    #: width of each image in pixels
    img_width: int
    #: number of images inside this one SER file
    num_frames: int

    _file: BinaryIO
    _buf: mmap.mmap

    _offset_array: np.ndarray
    _tag_offset_array: np.ndarray

    def open(self):
        """
        Open the SER file and populate metadata fields.
        This method does not read frame contents.
        """
        if sys.byteorder != 'little':
            raise OSError('SER reader expects little endian host')

        self._file = open(self.path, 'rb')
        self._buf = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

        # https://www3.ntu.edu.sg/home/cbb/info/TIAformat/index.html
        # read header
        byte_order, series_id, series_version, data_type_id, tag_type_id, total_num_elements, self.num_frames = struct.unpack(
            '<HHHIIII', self._buf[:22])
        assert byte_order == 0x4949
        assert series_id == 0x0197
        assert data_type_id == 0x4122, "only 2d array supported"
        assert tag_type_id in (0, 0x4142, 0x4152), "unknown tag type"

        if tag_type_id == 0:
            # empty file
            return

        varint = 'I' if series_version <= 0x0210 else 'Q'
        varint_len = 4 if series_version <= 0x0210 else 8
        varint_dtype = np.uint32 if series_version <= 0x0210 else np.uint64
        offset_array_offset = struct.unpack(f'<{varint}', self._buf[22:22 + varint_len])[0]

        # skip Dimension structs, they don't contain any useful information

        offset_array_len = total_num_elements * varint_len
        self._offset_array = np.frombuffer(self._buf[offset_array_offset:offset_array_offset + offset_array_len],
                                           dtype=varint_dtype)
        self._tag_offset_array = np.frombuffer(
            self._buf[offset_array_offset + offset_array_len:offset_array_offset + offset_array_len * 2],
            dtype=varint_dtype)

        # read height and width of the first image
        header_offset = self._offset_array[0]
        self.img_width, self.img_height = struct.unpack('<II', self._buf[header_offset + 42:header_offset + 50])

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

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def read_frame(self, idx: int) -> np.ndarray:
        """
        Extract the content of one of the images inside this SER file.
        :param idx: 0-indexed frame number
        :return: 2D int32 array
        """
        header_offset = self._offset_array[idx]
        data_type, array_size_x, array_size_y = struct.unpack('<HII', self._buf[header_offset + 40:header_offset + 50])
        assert data_type == 6, "only int32 images supported"

        img_len_bytes = 4 * array_size_y * array_size_x
        img_start_pos = header_offset + 50
        img = np.frombuffer(self._buf[img_start_pos:img_start_pos + img_len_bytes], dtype=np.int32).reshape(
            (array_size_y, array_size_x))
        return img

    def read_last_frame(self) -> np.ndarray:
        """
        Extract the content of the last frame.
        :return: 2D int32 array
        """
        return self.read_frame(self.num_frames - 1)

    def read_all_frames(self) -> list[np.ndarray]:
        """
        Extract the content of all the frames.
        :return: list of 2D int32 arrays
        """
        return [self.read_frame(i) for i in range(self.num_frames)]

    def read_timestamp(self, idx: int) -> int:
        """
        Read UNIX timestamp saved inside this file.
        :return: integer seconds since Jan 1, 1970, UTC.
        """
        tag_offset = self._tag_offset_array[idx]
        # includes 2 byte tag ID, 2 null bytes, then Unix timestamp:
        # 52 41 00 00 84 74 E4 66
        timestamp = struct.unpack('<I', self._buf[tag_offset + 4:tag_offset + 8])[0]
        return timestamp
