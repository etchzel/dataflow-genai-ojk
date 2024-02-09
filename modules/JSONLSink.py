from apache_beam.io import FileBasedSink
from apache_beam.io.filesystem import CompressionTypes

class _JSONLSink(FileBasedSink):
  def __init__(self, 
                file_path_prefix, 
                coder, 
                file_name_suffix='', 
                num_shards=0, 
                shard_name_template=None, 
                mime_type='application/json', 
                compression_type=CompressionTypes.AUTO, 
                *, 
                max_records_per_shard=None, 
                max_bytes_per_shard=None, 
                skip_if_empty=False):
    super(_JSONLSink, self).__init__(
      file_path_prefix, 
      coder, 
      file_name_suffix, 
      num_shards, 
      shard_name_template, 
      mime_type, 
      compression_type, 
      max_records_per_shard=max_records_per_shard, 
      max_bytes_per_shard=max_bytes_per_shard, 
      skip_if_empty=skip_if_empty
    )
    self.last_rows = dict()
  
  def open(self, temp_path):
    file_handle = super(_JSONLSink, self).open(temp_path)
    return file_handle

  def write_record(self, file_handle, value):
    import json

    if self.last_rows.get(file_handle, None) is not None:
        file_handle.write(self.coder.encode(json.dumps(self.last_rows[file_handle])))
        file_handle.write(self.coder.encode('\n'))

    self.last_rows[file_handle] = value

  def close(self, file_handle):
    import json

    if file_handle is not None:
        file_handle.write(self.coder.encode(json.dumps(self.last_rows[file_handle])))
        file_handle.write(self.coder.encode('\n'))
        file_handle.close()