"""
DataLoader pour ChairV2 dataset
PLUSIEURS sketches par photo (format: photo_id_N.png)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
import re

# ============================================================
# DATASET AVEC MULTI-SKETCHES
# ============================================================

class ChairV2Dataset(Dataset):
    """
    Dataset pour paires sketch-photo ChairV2
    
    Format des noms:
    - Sketch: 2429245009_1.png, 2429245009_2.png, 2429245009_20.png
    - Photo:  2429245009.png
    
    Un photo peut avoir N sketches différents
    """
    
    def __init__(
        self,
        root_dir,
        split='train',  # 'train' ou 'test'
        resolution=128,
        augment=True
    ):
        """
        Args:
            root_dir: Chemin vers ChairV2/
            split: 'train' ou 'test'
            resolution: Taille des images (128 ou 256)
            augment: Data augmentation (seulement pour train)
        """
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.resolution = resolution
        self.augment = augment and (split == 'train')
        
        # Définir dossiers (ignorer *_noise)
        if split == 'train':
            self.sketch_dir = self.root_dir / 'trainA'
            self.photo_dir = self.root_dir / 'trainB'
        else:
            self.sketch_dir = self.root_dir / 'testA'
            self.photo_dir = self.root_dir / 'testB'
        
        # Vérifier existence
        if not self.sketch_dir.exists():
            raise ValueError(f"Dossier sketch introuvable: {self.sketch_dir}")
        if not self.photo_dir.exists():
            raise ValueError(f"Dossier photo introuvable: {self.photo_dir}")
        
        # Lister fichiers
        self.sketch_files = sorted(self.sketch_dir.glob('*.png')) + \
                           sorted(self.sketch_dir.glob('*.jpg'))
        self.photo_files = sorted(self.photo_dir.glob('*.png')) + \
                          sorted(self.photo_dir.glob('*.jpg'))
        
        # Matcher sketches et photos
        self.pairs = self._match_pairs()
        
        print(f"📊 ChairV2 Dataset ({split}):")
        print(f"   Sketches fichiers: {len(self.sketch_files)}")
        print(f"   Photos fichiers:   {len(self.photo_files)}")
        print(f"   Paires créées:     {len(self.pairs)}")
        
        # Stats
        self._print_stats()
        
        # Transforms
        self._setup_transforms()
    
    def _extract_photo_id(self, filename):
        """
        Extrait l'ID de la photo depuis le nom de fichier
        
        Examples:
            2429245009_1.png  → 2429245009
            2429245009_20.png → 2429245009
            2429245009.png    → 2429245009
        """
        stem = filename.stem  # Sans extension
        
        # Enlever le suffixe _N si présent
        match = re.match(r'^(.+?)(?:_\d+)?$', stem)
        if match:
            return match.group(1)
        return stem
    
    def _match_pairs(self):
        """
        Matcher chaque sketch à sa photo correspondante
        
        Returns:
            List[(sketch_path, photo_path)]
        """
        # Créer mapping photo_id → photo_path
        photo_dict = {}
        for photo_path in self.photo_files:
            photo_id = self._extract_photo_id(photo_path)
            photo_dict[photo_id] = photo_path
        
        # Pour chaque sketch, trouver sa photo
        pairs = []
        unmatched = []
        
        for sketch_path in self.sketch_files:
            photo_id = self._extract_photo_id(sketch_path)
            
            if photo_id in photo_dict:
                pairs.append((sketch_path, photo_dict[photo_id]))
            else:
                unmatched.append(sketch_path.name)
        
        if unmatched:
            print(f"   ⚠️  {len(unmatched)} sketches sans photo correspondante")
            if len(unmatched) <= 10:
                print(f"      Exemples: {unmatched}")
        
        return pairs
    
    def _print_stats(self):
        """Affiche statistiques du dataset"""
        
        # Compter combien de sketches par photo
        from collections import Counter
        
        photo_counts = Counter()
        for sketch_path, photo_path in self.pairs:
            photo_counts[photo_path] += 1
        
        sketches_per_photo = list(photo_counts.values())
        
        print(f"\n   📈 Statistiques sketches par photo:")
        print(f"      Photos uniques:     {len(photo_counts)}")
        print(f"      Min sketches/photo: {min(sketches_per_photo)}")
        print(f"      Max sketches/photo: {max(sketches_per_photo)}")
        print(f"      Moyenne:            {np.mean(sketches_per_photo):.1f}")
        
        # Distribution
        unique_counts = sorted(set(sketches_per_photo))
        print(f"\n   📊 Distribution:")
        for count in unique_counts[:10]:  # Top 10
            n_photos = sum(1 for c in sketches_per_photo if c == count)
            print(f"      {count} sketch(s):  {n_photos} photos")
        
        if len(unique_counts) > 10:
            print(f"      ... et {len(unique_counts)-10} autres valeurs")
    
    def _setup_transforms(self):
        """Setup des transformations"""
        
        # Transform pour sketches (grayscale)
        sketch_transforms = [
            transforms.Resize((self.resolution, self.resolution)),
        ]
        
        if self.augment:
            sketch_transforms.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),  # 5 → 10
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.08, 0.08),   # 0.05 → 0.08
                    scale=(0.90, 1.10)        # 0.95/1.05 → 0.90/1.10
                ),
                # Nouveau: simuler différents styles de trait
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
                ], p=0.3),
                # Nouveau: variations de contraste du trait
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.15, contrast=0.3)
                ], p=0.4),
            ])
        
        sketch_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        self.sketch_transform = transforms.Compose(sketch_transforms)
        
        # Transform pour photos (RGB)
        photo_transforms = [
            transforms.Resize((self.resolution, self.resolution)),
        ]
        
        if self.augment:
            photo_transforms.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,   # 0.1 → 0.2
                    contrast=0.2,     # 0.1 → 0.2
                    saturation=0.2,   # 0.1 → 0.2
                    hue=0.08          # 0.05 → 0.08
                ),
                # Nouveau: légère perspective pour varier les angles
                transforms.RandomApply([
                    transforms.RandomPerspective(distortion_scale=0.1, p=1.0)
                ], p=0.2),
            ])
        
        photo_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.photo_transform = transforms.Compose(photo_transforms)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        sketch_path, photo_path = self.pairs[idx]
        
        # Charger sketch
        sketch = Image.open(sketch_path).convert('L')  # Grayscale
        sketch = self.sketch_transform(sketch)
        
        # Charger photo
        photo = Image.open(photo_path).convert('RGB')
        photo = self.photo_transform(photo)
        
        # Extraire photo_id
        photo_id = self._extract_photo_id(photo_path)
        
        return {
            'sketch': sketch,           # (1, H, W) range [-1, 1]
            'photo': photo,             # (3, H, W) range [-1, 1]
            'photo_id': photo_id,       # ID unique de la photo
            'sketch_path': str(sketch_path),
            'photo_path': str(photo_path),
        }

# ============================================================

# ============================================================
# DATASET PHOTO-ONLY (pour Photo Mapper)
# ============================================================

class PhotoOnlyDataset(Dataset):
    """
    Dataset avec SEULEMENT les photos (pas de duplicates)
    Pour entraîner le Photo Mapper (Phase 1)
    """
    
    def __init__(
        self,
        root_dir,
        split='train',
        resolution=128,
        augment=True
    ):
        self.root_dir = Path(root_dir)
        self.resolution = resolution
        self.augment = augment and (split == 'train')
        
        # Dossier photos
        if split == 'train':
            self.photo_dir = self.root_dir / 'trainB'
        else:
            self.photo_dir = self.root_dir / 'testB'
        
        if not self.photo_dir.exists():
            raise ValueError(f"Dossier photo introuvable: {self.photo_dir}")
        
        # Lister fichiers (photos uniques)
        self.photo_files = sorted(self.photo_dir.glob('*.png')) + \
                          sorted(self.photo_dir.glob('*.jpg'))
        
        print(f"📊 Photo-Only Dataset ({split}):")
        print(f"   Photos uniques: {len(self.photo_files)}")
        
        # Transform
        photo_transforms = [
            transforms.Resize((self.resolution, self.resolution)),
        ]
        
        if self.augment:
            photo_transforms.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.05
                ),
            ])
        
        photo_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.photo_transform = transforms.Compose(photo_transforms)
    
    def __len__(self):
        return len(self.photo_files)
    
    def __getitem__(self, idx):
        photo_path = self.photo_files[idx]
        
        photo = Image.open(photo_path).convert('RGB')
        photo = self.photo_transform(photo)
        
        return {
            'photo': photo,
            'photo_path': str(photo_path),
        }

# ============================================================
# AUGMENTATION PARTIELLE
# ============================================================

class PartialSketchAugmentation:
    """
    Augmentation pour sketches partiels
    Selon le papier: 30-100% du sketch, par paliers de 10%
    """
    
    def __init__(self, min_completion=0.3, max_completion=1.0, step=0.1):
        self.completions = np.arange(min_completion, max_completion + step, step)
    
    def __call__(self, sketch):
        """
        Rend le sketch partiellement visible
        
        Args:
            sketch: (1, H, W) tensor
        
        Returns:
            sketch_partial: (1, H, W) tensor
            completion_ratio: float (0.3 - 1.0)
            num_steps: int (3-10) nombre de latents à prédire
        """
        # Choisir ratio aléatoire
        completion = np.random.choice(self.completions)
        
        # Calculer nombre de steps (3-10 pour 30%-100%)
        # 30% → 3 latents, 100% → 10 latents
        num_steps = int(3 + (completion - 0.3) / 0.7 * 7)
        num_steps = np.clip(num_steps, 3, 10)
        
        if completion >= 1.0:
            return sketch, completion, num_steps
        
        # Masquer une portion du sketch
        # Stratégie: masquer verticalement (de bas en haut)
        H = sketch.size(-2)
        visible_height = int(H * completion)
        
        sketch_partial = sketch.clone()
        sketch_partial[:, visible_height:, :] = 1.0  # Blanc (background)
        
        return sketch_partial, completion, num_steps

# ============================================================
# ============================================================
# TRIPLET DATASET - VERSION CORRIGÉE
# ============================================================

class TripletChairDataset(Dataset):
    """
    Dataset pour FG-SBIR avec triplets
    (anchor_sketch, positive_photo, negative_photo)
    """
    
    def __init__(self, root_dir, split='train', resolution=128):
        """
        Args:
            root_dir: Chemin vers ChairV2/
            split: 'train' ou 'test'
            resolution: Taille des images
        """
        # ✅ Utiliser ChairV2Dataset directement
        self.base_dataset = ChairV2Dataset(
            root_dir=root_dir,
            split=split,
            resolution=resolution,
            augment=(split == 'train')
        )
        
        # ✅ Créer mapping photo_id → indices
        # photo_id est déjà dans le dict retourné par ChairV2Dataset
        self.photo_to_indices = {}
        
        for idx in range(len(self.base_dataset)):
            # ✅ Accéder via __getitem__ pour obtenir photo_id
            item = self.base_dataset[idx]
            photo_id = item['photo_id']
            
            if photo_id not in self.photo_to_indices:
                self.photo_to_indices[photo_id] = []
            self.photo_to_indices[photo_id].append(idx)
        
        self.all_photo_ids = list(self.photo_to_indices.keys())
        
        print(f"   📊 Triplet Dataset ({split}):")
        print(f"      Total paires: {len(self.base_dataset)}")
        print(f"      Photos uniques: {len(self.all_photo_ids)}")
        print(f"      Sketches par photo (moy): {len(self.base_dataset) / len(self.all_photo_ids):.1f}")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # ✅ Anchor + Positive
        anchor_item = self.base_dataset[idx]
        anchor_sketch = anchor_item['sketch']
        positive_photo = anchor_item['photo']
        anchor_photo_id = anchor_item['photo_id']  # ✅ Déjà extrait
        
        # ✅ Negative: photo d'un autre ID
        negative_photo_id = np.random.choice([
            pid for pid in self.all_photo_ids if pid != anchor_photo_id
        ])
        
        negative_idx = np.random.choice(self.photo_to_indices[negative_photo_id])
        negative_item = self.base_dataset[negative_idx]
        negative_photo = negative_item['photo']
        
        return {
            'anchor_sketch': anchor_sketch,
            'positive_photo': positive_photo,
            'negative_photo': negative_photo,
        }

print("✅ TripletChairDataset corrigé défini")