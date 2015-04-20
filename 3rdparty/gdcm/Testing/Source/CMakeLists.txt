# All the tests

# ADD_SUBDIRECTORY is required here to properly get the definition variables later on:
add_subdirectory(Data)
# Get the variables defined in Data subdirs:
get_directory_property(gdcm_data_dicomdir_filenames_glob DIRECTORY Data DEFINITION GDCM_DATA_DICOMDIR_FILENAMES_GLOB)
get_directory_property(gdcm_data_image_filenames_glob DIRECTORY Data DEFINITION GDCM_DATA_IMAGE_FILENAMES_GLOB)
get_directory_property(gdcm_data_filenames_glob DIRECTORY Data DEFINITION GDCM_DATA_FILENAMES_GLOB)
get_directory_property(black_list_reader DIRECTORY Data DEFINITION BLACK_LIST_READER)

subdirs(
  Attribute
  Common
  DataDictionary
  DataStructureAndEncodingDefinition
  InformationObjectDefinition
  MediaStorageAndFileFormat
  MessageExchangeDefinition
)
