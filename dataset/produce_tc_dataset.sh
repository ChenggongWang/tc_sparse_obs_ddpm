#!/bin/bash  

set -e #end with any error
#set -x #expands variables and prints a little + sign before the line
# load modules or conda environments here
source /usr/share/Modules/init/bash
module purge
module load anaconda3/2023.3
source /usr/licensed/anaconda3/2023.3/etc/profile.d/conda.sh
conda activate cg310 

for i in {120..130}
do
    echo "year: $i"
    python -u data_tc.py WVP    $i
#     python -u data_tc.py slp    $i 
#     python -u data_tc.py v_ref  $i 
#     python -u data_tc.py u_ref  $i 
#     python -u data_tc.py precip $i 
done
echo done