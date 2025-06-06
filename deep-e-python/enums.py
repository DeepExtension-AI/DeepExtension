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
from enum import Enum
class LogEnum(Enum):
    ## Common logs for other internal services
    ParamCheckFailed    = 120001
    ParamCheckWarning   = 120002
    StartHandlingEvents = 120003
    HandleEventsSuccess = 120004
    HandleEventsFailed  = 120005

    ## Fine-tuning related
    TrainingFileNotFound     = 130001
    LoadingTrainingFileError = 130002

    ## Other related
    ActualParams = 140001

    ## Training file internal logs
    InitializingModel       = 200001
    HandleWithDataset       = 200002
    ConfigTrainingParams    = 200003
    StartTraining           = 200004
    TrainingSuccess         = 200005
    TrainingFailed          = 200006
    SaveTrainedModel        = 200007
    SaveTrainedModelSuccess = 200008
    Training                = 200009

    ## Model saving related
    SaveModelTaskStarts       = 500001
    SaveModelTaskSuccess      = 500002
    SaveModelTaskFailed       = 500003
    StartSaveModelTaskService = 500004
    SaveModelResponseError    = 500005
    SaveModelResponseSuccess  = 500006
    MergeLoadingBaseModel     = 500007
    MergeModelQuantizing      = 500008
    MergeSuccess              = 500009
    MergeAdapters             = 500010

    ## Deployment related
    DeployInitializeOllama       = 600001
    DeployCalculateSha256Start   = 600002
    DeployCalculateSha256Success = 600003
    DeployCalculateSha256Failed  = 600004
    DeployUploadingFiles         = 600005
    DeployUploadFileSuccess      = 600006
    DeployUploadFileSkip         = 600007
    DeployUploadFileFailed       = 600008
    DeployUploadFileError        = 600009
    DeployUploadingFolders       = 600010
    DeployUploadFoldersSuccess   = 600011
    DeployUploadFoldersFailed    = 600012
    DeployGenerateHash           = 600013
    DeployGenerateHashSuccess    = 600014
    DeployGenerateHashSkip       = 600015
    DeployHandleWithHash         = 600016
    DeployCreateModelStart       = 600017
    DeployCreateModelRequest     = 600018
    DeployEnableQuantize         = 600019
    DeploySendingCreateModel     = 600020
    DeployCreateStatusCode       = 600021
    DeployCreateModelSuccess     = 600022
    DeployCreateModelFailed      = 600023
    DeployCreateModelRawResponse = 600024
    DeployNetworkError           = 600025
    DeployError                  = 600026
    DeployNotFullSuccess         = 600027
    DeployUploadFoldersStart     = 600028
    DeployOllamaError       = 600029

class StatusEnum(Enum):
    PleaseCheck = 0
    Success    = 1
    Running    = 2
    Failed     = 3
    Warning    = 4
    Unknown    = 5

class LevelEnum(Enum):
    INFO    = 0
    WARNING = 1
    ERROR   = 2