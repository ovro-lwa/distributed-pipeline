import sys
from gpu_subband_imaging.worker import _mjd
from gpu_subband_imaging.stages import flag
stamp = sys.argv[1]
v = flag.badants(_mjd(stamp), "/opt/devel/pipeline/envs/development/bin/python")
print(stamp, _mjd(stamp), v)
open(sys.argv[2], "w").write(v + "\n")
