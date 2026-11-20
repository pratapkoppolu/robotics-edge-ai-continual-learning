"""Register a RAW experimental checkpoint as an MLflow model-version artifact.
Not an MLmodel flavor, cannot use mlflow.pyfunc.load_model and NEVER a release.
Opt-in only: logging large .pt files duplicates storage.
"""
import argparse, hashlib, json
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def register(manifest_path, tracking_uri='sqlite:///mlflow.db', confirm=False):
    manifest_path=Path(manifest_path)
    m=json.loads(manifest_path.read_text(encoding='utf-8'))
    cp=Path(m['artifact'])
    if not cp.exists(): raise FileNotFoundError(cp)
    if digest(cp)!=m['artifact_sha256']: raise ValueError('Checkpoint checksum mismatch')
    if not (m.get('simulation_only') is True and m.get('approved_for_deployment') is False):
        raise ValueError('This experimental registry utility requires simulation_only=true and deployment=false')
    name=f"robotics-student-{str(m['task']).lower()}-experimental-raw"
    plan={'model_name':name,'checkpoint':str(cp),'size_mb':round(cp.stat().st_size/1e6,1),'simulate_only':True,'would_duplicate_checkpoint':True}
    if not confirm: return {'mode':'dry_run',**plan}
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment('robotics_registry_experimental')
    with mlflow.start_run(run_name=m['candidate_version']) as run:
        mlflow.set_tags({'simulation_only':'true','deployable':'false','artifact_format':'raw-pytorch-checkpoint','candidate_version':m['candidate_version'],'checkpoint_sha256':m['artifact_sha256']})
        mlflow.log_artifact(str(cp),artifact_path='raw_checkpoint')
        mlflow.log_artifact(str(manifest_path),artifact_path='provenance')
        artifact_uri=mlflow.get_artifact_uri(f'raw_checkpoint/{cp.name}')
        client=MlflowClient()
        try: client.create_registered_model(name)
        except mlflow.exceptions.MlflowException as exc:
            if 'already exists' not in str(exc).lower(): raise
        version=client.create_model_version(name=name,source=artifact_uri,run_id=run.info.run_id,
            tags={'simulation_only':'true','deployable':'false','format':'raw-checkpoint','sha256':m['artifact_sha256']})
        return {'mode':'registered_raw_artifact','registered_model_name':name,'version':version.version,'run_id':run.info.run_id,'source':artifact_uri,'deployable':False,'mlmodel_flavor':False}


def main():
    a=argparse.ArgumentParser();a.add_argument('--manifest',default='outputs/step12_task_b/candidate_manifest.json');a.add_argument('--tracking-uri',default='sqlite:///mlflow.db');a.add_argument('--confirm-copy-and-register',action='store_true');a.add_argument('--out',default='outputs/step12_task_b/registry_step13.json');args=a.parse_args()
    result=register(args.manifest,args.tracking_uri,args.confirm_copy_and_register)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__': main()
