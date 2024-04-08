import numpy as np
import cv2


spec_dir = 'results/data/minion/000000_0_spec.jpg'
diff_dir = 'results/data/minion/diff.jpg'

spec = cv2.imread(spec_dir) / 255.
diff = cv2.imread(diff_dir) / 255.

# import ipdb; ipdb.set_trace()
spec_div_diff = spec / (diff + 1e-6)

cv2.imwrite('test_spec.png', (np.hstack([spec, diff, np.abs(spec_div_diff.clip(0,1)-spec)]) * 255).astype(np.uint8))
