~/projects/LPCNet/convert_cepstrum_cxx $1 /tmp/output.pcm
cat /tmp/output.pcm | nc 10.224.2.125 11234
