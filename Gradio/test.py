import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True) # th, tts, wav2lip
parser.add_argument('--gpus', type=str, default='4,5,6,7') 
parser.add_argument('--twin_id', type=str, required=True) 
parser.add_argument('--image', type=str, required=True) 
parser.add_argument('--video', type=str, required=True)
parser.add_argument('--output_name', type=str, required=True)
parser.add_argument('--preprocess_drv', type=int, default=1)

args, unparsed  = parser.parse_known_args()
if len(unparsed) != 0:
    raise SystemExit('Unknown argument: {}'.format(unparsed))

if args.model == 'tts':
	sys.path.insert(0,'models/TTS')
	from pipelines.tts_test import TTSTest
	TTSTest(args.gpus)

elif args.model == 'wav2lip':
	sys.path.insert(0,'models/Wav2Lip')
	from pipelines.wav2lip_test import Wav2LipTest
	Wav2LipTest(args.gpus)

elif args.model == 'th':
	sys.path.insert(0,'models/TH')
	from pipelines.th_test import THTest
	THTest(args.gpus, twin_id=args.twin_id, test_image=args.image, test_video=args.video, output_name=args.output_name, preprocess_drv=args.preprocess_drv)
