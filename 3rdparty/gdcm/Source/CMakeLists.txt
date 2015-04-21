# ...
subdirs(
  Common
  Attribute
  DataDictionary
  DataStructureAndEncodingDefinition
  InformationObjectDefinition
  MediaStorageAndFileFormat  
  )

if(GDCM_USE_MEXD_MODULE)
  subdirs(MessageExchangeDefinition)
endif()