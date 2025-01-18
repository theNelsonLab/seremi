# Example how to read metadata from EMI and frame content from SER

from argparse import ArgumentParser
from seremi import EMIFile, SERFile
import numpy as np

parser = ArgumentParser()
parser.add_argument('--ser', required=True, help='path to SER file, such as movie_1.ser')
parser.add_argument('--emi', required=True, help='path to EMI file, such as movie.emi')
args = parser.parse_args()

with SERFile(args.ser) as ser, EMIFile(args.emi) as emi:
    # EMI contains detailed metadata, mostly in info_dict
    voltage = emi.info_dict['ObjectInfo']['ExperimentalConditions']['MicroscopeConditions']['AcceleratingVoltage']
    print(f'voltage={voltage}')

    # read image content from SER
    for i in range(ser.num_frames):
        frame = ser.read_frame(i)  # numpy array
        print(f'frame {i} average: {np.mean(frame)}')  # do something with the frame content

    # EMI file contains the last frame of the SER
    assert np.array_equal(emi.read_frame(), ser.read_last_frame())
