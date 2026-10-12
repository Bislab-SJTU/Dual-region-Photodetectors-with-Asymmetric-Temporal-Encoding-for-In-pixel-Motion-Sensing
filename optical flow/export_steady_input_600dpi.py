from pathlib import Path
import hashlib,json,sys
import numpy as np
import matplotlib
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(r'D:\石阳\博士工作\科研\周报\week update-140-shiyang')
EXP=ROOT/'Farneback_稳态响应_亮度变化'
OLD=ROOT.parent/'week update-132-shiyang/simulation/result/亮度变化/稳态响应/frames'
ARRAY=ROOT/'simulation_连续动态模型/result/亮度变化/稳态响应/array_current_A.npy'
OUT=EXP/'02_输入与参数/原始帧图_600dpi';OUT.mkdir(parents=True,exist_ok=True)
cube=np.load(ARRAY,mmap_mode='r');cmap=matplotlib.colormaps['Blues']
records=[]
for fi,p in enumerate(sorted(OLD.glob('frame_*.png'))):
    rgb=(cmap(np.clip(np.flipud(np.abs(cube[:,:,fi*200]))*1e6/135.,0,1))[:,:,:3]*255).astype('uint8')
    raster=Image.fromarray(rgb)
    old=Image.open(p).convert('RGB')
    check=np.array(raster.resize(old.size,Image.Resampling.NEAREST))
    assert np.array_equal(check,np.array(old)),f'Current data differs from original frame {fi}'
    dest=OUT/p.name
    # Render directly from the 100x100 array, without interpolated intermediate values.
    raster.resize((3600,2168),Image.Resampling.NEAREST).save(dest,dpi=(600,600))
    with Image.open(dest) as im:
        assert im.size==(3600,2168) and abs(im.info['dpi'][0]-600)<.1
    records.append({'frame':fi,'sample_index':fi*200,'original_png_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'new_png_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'matches_original_at_original_resolution':True})
assert len(records)==79
(EXP/'02_输入与参数/原始帧图_600dpi_检查.json').write_text(json.dumps({'source_array':str(ARRAY),'reference_frames':str(OLD),'width':3600,'height':2168,'dpi':600,'mapping':'abs current; Blues 0..135uA; nearest-neighbor, no smoothing','purpose':'high-resolution display/export; existing Farneback was computed at original 332x200 resolution, not recalculated','physical_information':'100x100 array; higher export resolution does not add measured detail','frames':records},ensure_ascii=False,indent=2),encoding='utf-8')
print('Verified and exported 79 frames: 3600x2168, 600dpi')
