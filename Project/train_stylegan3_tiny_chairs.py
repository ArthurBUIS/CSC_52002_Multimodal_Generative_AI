# train_stylegan3_final_fixed.py
"""
Version finale qui fonctionne - SANS expandable_segments
"""

import os
import sys
import warnings
from PIL import Image
warnings.filterwarnings('ignore')

print("🔧 Configuration (mode TINY)...")

# Environnement - SANS expandable_segments
os.environ['LD_LIBRARY_PATH'] = ''
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.9.1'
os.environ['PYTORCH_JIT'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # ← RETIRÉ expandable_segments

import torch
print(f"✅ PyTorch {torch.__version__}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("✅ Cache CUDA vidé")

# Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
stylegan3_path = os.path.join(script_dir, 'stylegan3')

if not os.path.isdir(stylegan3_path):
    raise FileNotFoundError(
        f"Dossier stylegan3 introuvable: {stylegan3_path}\n"
        "Clonez-le d'abord dans le dossier Project: git clone https://github.com/NVlabs/stylegan3.git"
    )

sys.path.insert(0, stylegan3_path)
os.chdir(stylegan3_path)
dataset_path = os.path.join(script_dir, 'filtered_images')


def prepare_stylegan_dataset(input_dir, resolution=128):
    output_dir = os.path.join(script_dir, f'filtered_images_stylegan_{resolution}')
    os.makedirs(output_dir, exist_ok=True)

    valid_ext = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    converted = 0
    skipped = 0

    for filename in sorted(os.listdir(input_dir)):
        if not filename.lower().endswith(valid_ext):
            continue
        src = os.path.join(input_dir, filename)
        stem, _ = os.path.splitext(filename)
        dst = os.path.join(output_dir, f'{stem}.png')

        try:
            with Image.open(src) as img:
                img = img.convert('RGB')
                img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)
                img.save(dst, format='PNG')
            converted += 1
        except Exception:
            skipped += 1

    if converted == 0:
        raise RuntimeError(
            f"Aucune image valide trouvée dans {input_dir}. "
            "Vérifiez le dossier et les extensions."
        )

    print(f"✅ Dataset préparé: {output_dir}")
    print(f"   Images converties: {converted}")
    if skipped > 0:
        print(f"   Images ignorées (erreur): {skipped}")
    return output_dir

# Patches
print("\n🔨 Patches...")
patches = {
    'torch_utils/ops/bias_act.py': [
        ("impl='cuda'", "impl='ref'"),
        ("impl = 'cuda'", "impl = 'ref'"),
    ],
    'torch_utils/ops/upfirdn2d.py': [
        ("impl='cuda'", "impl='ref'"),
        ("impl = 'cuda'", "impl = 'ref'"),
    ],
    'torch_utils/ops/conv2d_gradfix.py': [("enabled = _should_use_custom_op", "enabled = False #")],
    'torch_utils/ops/filtered_lrelu.py': [
        ("impl='cuda'", "impl='ref'"),
        ("impl = 'cuda'", "impl = 'ref'"),
    ],
}

patched_files = 0
for f, reps in patches.items():
    if os.path.exists(f):
        with open(f) as fp:
            content = fp.read()
        original = content
        for old, new in reps:
            content = content.replace(old, new)
        with open(f, 'w') as fp:
            fp.write(content)
        if content != original:
            patched_files += 1

print(f"✅ Fichiers patchés: {patched_files}/{len(patches)}")

# Imports
print("\n📦 Imports...")
import dnnlib
from training import training_loop
import legacy

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {props.total_memory / 1e9:.1f} GB")

# Configuration
print("\n⚙️  Configuration...")

dataset_path = prepare_stylegan_dataset(dataset_path, resolution=128)

c = dnnlib.EasyDict()

c.training_set_kwargs = dnnlib.EasyDict(
    class_name='training.dataset.ImageFolderDataset',
    path=dataset_path, # dossier local contenant les images .jpg des chaises
    use_labels=False,
    max_size=None,
    xflip=True,
    resolution=128,
    random_seed=0
)

c.data_loader_kwargs = dnnlib.EasyDict(
    pin_memory=False,
    num_workers=0
)

c.G_kwargs = dnnlib.EasyDict(
    class_name='training.networks_stylegan2.Generator',
    z_dim=256,
    w_dim=256,
    mapping_kwargs=dnnlib.EasyDict(num_layers=4),
    channel_base=8192,
    channel_max=256,
)

c.D_kwargs = dnnlib.EasyDict(
    class_name='training.networks_stylegan2.Discriminator',
    block_kwargs=dnnlib.EasyDict(freeze_layers=0),
    mapping_kwargs=dnnlib.EasyDict(),
    epilogue_kwargs=dnnlib.EasyDict(mbstd_group_size=2),
    channel_base=8192,
    channel_max=256,
)

c.G_opt_kwargs = dnnlib.EasyDict(
    class_name='torch.optim.Adam',
    betas=[0, 0.99],
    eps=1e-8,
    lr=0.002
)

c.D_opt_kwargs = dnnlib.EasyDict(
    class_name='torch.optim.Adam',
    betas=[0, 0.99],
    eps=1e-8,
    lr=0.002
)

c.loss_kwargs = dnnlib.EasyDict(
    class_name='training.loss.StyleGAN2Loss',
    r1_gamma=1.0,
    style_mixing_prob=0.0,
    pl_weight=0,
    pl_no_weight_grad=True
)

c.num_gpus = 1
c.batch_size = 8
c.batch_gpu = 8
c.total_kimg = 64000  # Correspond à ce qui est écrit dans le papier, en pratique prend une semaine complète d'entraînement, donc 64000 kimg est un objectif ambitieux
c.kimg_per_tick = 4
c.ema_kimg = 10
c.G_reg_interval = 8
c.random_seed = 0
c.network_snapshot_ticks = 50
c.image_snapshot_ticks = 50
c.metrics = []
c.augment_kwargs = None
c.ada_target = None

c.run_dir = os.path.join(script_dir, 'stylegan3_training_chairs')
os.makedirs(c.run_dir, exist_ok=True)

print(f"✅ Configuration:")
print(f"   Résolution: 128x128")
print(f"   Dataset: {dataset_path}")
print(f"   Batch: {c.batch_size}")
print(f"   Kimg total: {c.total_kimg}")
print(f"   Output: {c.run_dir}")

print("\n" + "="*60)
print("🚀 DÉMARRAGE ENTRAÎNEMENT")
print("="*60 + "\n")

# ======= Checkpoint de reprise automatique (optionnel) =======
snapshots = sorted(
    [
        f for f in os.listdir(c.run_dir)
        if f.startswith('network-snapshot-') and f.endswith('.pkl')
    ]
)

resume_path = None
resume_kimg = 0

for snapshot in reversed(snapshots):
    candidate_path = os.path.join(c.run_dir, snapshot)
    try:
        with open(candidate_path, 'rb') as f:
            legacy.load_network_pkl(f)
        resume_path = candidate_path
        try:
            # network-snapshot-001200.pkl -> 1200 kimg
            resume_kimg = int(snapshot.split('-')[-1].split('.')[0])
        except ValueError:
            resume_kimg = 0
        break
    except Exception as err:
        print(f"⚠️  Snapshot ignoré (corrompu/invalide): {snapshot} ({type(err).__name__})")

if resume_path is not None:
    print(f"🔄 Reprise depuis {resume_path}")
    c.resume_pkl = resume_path
    c.resume_kimg = resume_kimg
elif snapshots:
    print("⚠️  Snapshots trouvés mais tous invalides, entraînement from scratch")
else:
    print("⚠️  Pas de checkpoint trouvé, entraînement from scratch")


try:
    training_loop.training_loop(rank=0, **c)
    
    print("\n" + "="*60)
    print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print(f"📁 Checkpoints: {c.run_dir}")
    print("="*60)
    
except KeyboardInterrupt:
    print("\n⚠️  Entraînement interrompu par l'utilisateur")
    print(f"📁 Checkpoints sauvegardés: {c.run_dir}")
    
except Exception as e:
    print(f"\n❌ {type(e).__name__}: {str(e)[:300]}")
    import traceback
    traceback.print_exc()
    print(f"\n📁 {c.run_dir}")