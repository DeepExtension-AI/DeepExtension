"""
 /*
  * Copyright 2025 DeepExtension team
  *
  * Licensed under the Apache License, Version 2.0 (the "License");
  * you may not use this file except in compliance with the License.
  * You may obtain a copy of the License at
  *
  *     http://www.apache.org/licenses/LICENSE-2.0
  *
  * Unless required by applicable law or agreed to in writing, software
  * distributed under the License is distributed on an "AS IS" BASIS,
  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  * See the License for the specific language governing permissions and
  * limitations under the License.
  */
"""
import json
import redis
from datetime import timedelta,datetime
import os
class RedisClient:
    def __init__(self):
        import platform
        system_type = platform.system().lower()
        global redis_host,redis_py_port
        if system_type == "linux":
            redis_host=os.getenv('AI_PY_REDIS_HOST') 
            redis_py_port=os.getenv('AI_PY_REDIS_START_PORT')
        elif system_type == "darwin": 
            redis_host="127.0.0.1"
            redis_py_port=os.getenv('AI_PY_REDIS_EXPOSED_PORT')
        print(f"Using redis host:{redis_host},redis port:{redis_py_port}")
        self.client = redis.Redis(
            host=redis_host,
            port=redis_py_port,
            db=0,
            decode_responses=True  
        )
    
    def set_status(self, task_id: str, status: int, pid: int = 0):
        from datetime import datetime, timezone
        # 使用 UTC 时区
        now_utc = datetime.now(timezone.utc)
        self.client.set(
            name=task_id,
            value=json.dumps({
                "status": status,
                "updateTime": now_utc.isoformat(),
                "pid": pid
            })
        )
    
    def get_status(self, task_id: str) -> dict:
        data = self.client.get(task_id)
        return json.loads(data) if data else None
    
    def delete_status(self, task_id: str):
        self.client.delete(task_id)


redis_client = RedisClient()
