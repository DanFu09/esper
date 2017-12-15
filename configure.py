import argparse
import yaml
import toml
import subprocess as sp
import shlex
from dotmap import DotMap
import multiprocessing

NGINX_PORT = '80'
IPYTHON_PORT = '8888'

cores = multiprocessing.cpu_count()
config = DotMap(
    yaml.load("""
version: '2'
services:
  nginx:
    image: nginx
    command: ["bash", "/tmp/subst.sh"]
    volumes:
      - ./app:/app
      - ./nginx:/tmp
    depends_on: [app, frameserver]
    ports: ["{nginx_port}:{nginx_port}"]
    environment: ["PORT={nginx_port}"]

  frameserver:
    image: scannerresearch/frameserver
    ports: ['7500']
    environment:
      - 'WORKERS={workers}'

  app:
    build:
      context: ./app
      args:
        https_proxy: "${{https_proxy}}"
        cores: {cores}
    privileged: true
    depends_on: [db]
    volumes:
      - ./app:/app
      - ${{HOME}}/.bash_history:/root/.bash_history
      - ./service-key.json:/app/service-key.json
    ports: ["8000", "{ipython_port}:{ipython_port}"]
    environment: ["IPYTHON_PORT={ipython_port}", "JUPYTER_PASSWORD=esperjupyter"]
""".format(nginx_port=NGINX_PORT, ipython_port=IPYTHON_PORT, cores=cores, workers=cores * 2)))

db_local = DotMap(
    yaml.load("""
image: postgres:9.5
environment:
  - POSTGRES_USER=will
  - POSTGRES_PASSWORD=foobar
  - POSTGRES_DB=esper
volumes: ["./postgres:/var/lib/postgresql/data"]
ports: ["5432"]
"""))

db_google = DotMap(
    yaml.load("""
image: gcr.io/cloudsql-docker/gce-proxy:1.09
volumes: ["./service-key.json:/config"]
environment: []
ports: ["5432"]
"""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True)
    parser.add_argument('--dataset', default='default')
    args = parser.parse_args()

    # TODO(wcrichto): validate config file
    base_config = DotMap(toml.load(args.config))

    if 'google' in base_config:
        config.services.app.environment.append('GOOGLE_PROJECT={}'.format(
            base_config.google.project))

    if 'compute' in base_config:
        if 'gpu' in base_config.compute:
            device = 'gpu' if base_config.compute.gpu else 'cpu'
        else:
            device = 'cpu'
    else:
        device = 'cpu'
    config.services.app.build.args.device = device
    config.services.app.build.args.device2 = device
    config.services.app.environment.append('DEVICE={}'.format(device))
    config.services.app.image = 'scannerresearch/esper:{}'.format(device)

    if base_config.database.type == 'google':
        assert 'google' in base_config
        config.services.db = db_google
        config.services.db['command'] = \
            '/cloud_sql_proxy -instances={project}:{zone}:{name}=tcp:0.0.0.0:5432 -credential_file=/config'.format(
                project=base_config.google.project, zone=base_config.google.zone, name=base_config.database.name)
    else:
        config.services.db = db_local

    config.services.app.environment.append('DJANGO_DB_USER={}'.format(
        base_config.database.user if 'user' in base_config.database else 'root'))

    if 'password' in base_config.database:
        config.services.app.environment.append("DJANGO_DB_PASSWORD={}".format(
            base_config.database.password))

    scanner_config = {'scanner_path': '/opt/scanner'}
    if base_config.storage.type == 'google':
        assert 'google' in base_config
        scanner_config['storage'] = {
            'type': 'gcs',
            'bucket': base_config.storage.bucket,
            'db_path': '{}/scanner_db'.format(base_config.storage.path)
        }
    else:
        scanner_config['storage'] = {'type': 'posix', 'db_path': '/app/scanner_db'}

    config.services.frameserver.environment.append('FILESYSTEM={}'.format(base_config.storage.type))

    for service in config.services.values():
        env_vars = [
            'ESPER_ENV={}'.format(base_config.storage.type), 'DATASET={}'.format(args.dataset),
            'DATA_PATH={}'.format(base_config.storage.path)
        ]

        if base_config.storage.type == 'google':
            env_vars.extend([
                'BUCKET={}'.format(base_config.storage.bucket),
                'AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}',
                'AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}',
            ])

        service.environment.extend(env_vars)

    with open('app/.scanner.toml', 'w') as f:
        f.write(toml.dumps(scanner_config))

    with open('docker-compose.yml', 'w') as f:
        f.write(yaml.dump(config.toDict()))

    print('Successfully configured Esper.')


if __name__ == '__main__':
    main()
