time_tag="-$(date +%F-%H-%M-%S)"
experiment_tag="default"
if [ -z "$1" ]
then
      echo "You did not designate a experiment tag, using [default]"
else
      echo "Your experiment tag is $1"
      experiment_tag = "$1"
fi


# create top dir
top_out_dir="output_log"

if [ ! -d "${top_out_dir}" ];then
       mkdir -p "${top_out_dir}"
fi



# create sub dir
sub_out_dir="${top_out_dir}/${experiment_tag}${time_tag}"

real_time_log="${sub_out_dir}/${experiment_tag}${time_tag}.txt"

echo "Your log will save in folder ${sub_out_dir}"

if [ ! -d ${sub_out_dir} ];then
       mkdir -p ${sub_out_dir}
fi

if [ -z "$2" ]
then
      echo "You did not designate a config file, using default one by policy."
      python simple_rl_train_with_detection.py -p td3  -n ${sub_out_dir} 2>&1|tee ${real_time_log}
else
      echo "Your config file is $2"
      python simple_rl_train_with_detection.py -p td3  -n ${sub_out_dir} -d $2 2>&1|tee ${real_time_log}
fi


