import carla
import yaml
import logging

from noisy_planning.config.ppo_config import *

servers = ppo_config['server']
active_num = 0
server_list = []
for i in servers:
    carla_host = i['carla_host']
    carla_ports = i['carla_ports']
    carla_ports = [i for i in range(carla_ports[0], carla_ports[1], carla_ports[2])]
    for port in carla_ports:
        server_list.append((carla_host, int(port)))
num = len(server_list)


logging.error("===========Active Server Checking of {} servers ==============".format(num))
for i in server_list:
    carla_host, carla_ports = i
    tmp_client = carla.Client(carla_host, carla_ports)
    try:
        tmp_client.get_world()
        logging.error("inx:{}, {}:{} is active...".format(i, carla_host, carla_ports))
        active_num += 1
        del tmp_client
    except Exception as e:
        logging.error("inx:{}, {}:{} is not active...".format(i, carla_host, carla_ports))
logging.error("Active serve num: {}/{}".format(active_num, num))
logging.error("===========Active Server Checking: Over ==============")
