import inspect
import lightning_fabric.utilities.cloud_io as cloud_io

print(inspect.getsource(cloud_io._atomic_save))
