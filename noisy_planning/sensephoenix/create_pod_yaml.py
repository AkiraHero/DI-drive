import yaml

sample = "/home/akira/Project/Model_behaviour/DI-drive/noisy_planning/carla0911pod.yaml"
outfile = "tst.yaml"
namespace = "shlab-cla"
name_app = "carla0911pod11"
app_label_name = name_app
service_name = name_app + '-service'
start_port = 30417
num = 5
container_name = name_app

with open(sample) as f:
    docs = yaml.safe_load_all(f)
    docs = [doc for doc in docs]
    pod_config = docs[0]
    svc_config = docs[1]
    pod_config['metadata']['namespace'] = namespace
    pod_config['metadata']['name'] = name_app
    pod_config['metadata']['labels']['app'] = app_label_name
    pod_config['spec']['containers'][0]['args'] = ["./start.sh -n {} -p {}".format(num, start_port)]
    pod_config['spec']['containers'][0]['name'] = container_name
    svc_config['metadata']['name'] = service_name
    svc_config['spec']['ports'] = []
    cur_port = start_port
    for i in range(num * 2):
        port_instance = {}
        port_instance['port'] = cur_port
        port_instance['targetPort'] = cur_port
        port_instance['nodePort'] = cur_port
        port_instance['name'] = "p{}".format(cur_port)
        port_instance['protocol'] = 'TCP'
        svc_config['spec']['ports'].append(port_instance)
        svc_config['spec']['selector']['app'] = app_label_name
        cur_port += 1

    target_configs = [pod_config, svc_config]
    with open(outfile, 'w') as fout:
        yaml.dump_all(target_configs, fout)
    pass

