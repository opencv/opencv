/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmCryptoFactory.h"
#include <string.h>
#include <stdio.h>
#include <memory>
#include <vector>

#include "gdcmFilename.h"
#include "gdcmTesting.h"

static bool LoadFile(const char * filename, char* & buffer, size_t & bufLen)
{
  FILE * f = fopen(filename, "rb");
  if (f == NULL)
    {
    //gdcmErrorMacro("Couldn't open the file: " << filename);
    return false;
    }
  fseek(f, 0L, SEEK_END);
  long sz = ftell(f);
  rewind(f);
  buffer = new char[sz];
  bufLen = sz;
  while (sz)
    sz -= fread(buffer + bufLen - sz, sizeof(char), sz, f);
  return true;
}

static const gdcm::CryptographicMessageSyntax::CipherTypes ciphers[] = {
  gdcm::CryptographicMessageSyntax::AES128_CIPHER,
  gdcm::CryptographicMessageSyntax::AES192_CIPHER,
  gdcm::CryptographicMessageSyntax::AES256_CIPHER,
  gdcm::CryptographicMessageSyntax::DES3_CIPHER
  };

static std::pair<gdcm::CryptographicMessageSyntax::CipherTypes, std::string> cip2str_data[] = {
    std::make_pair(gdcm::CryptographicMessageSyntax::AES128_CIPHER, "AES128"),
    std::make_pair(gdcm::CryptographicMessageSyntax::AES192_CIPHER, "AES192"),
    std::make_pair(gdcm::CryptographicMessageSyntax::AES256_CIPHER, "AES256"),
    std::make_pair(gdcm::CryptographicMessageSyntax::DES3_CIPHER,   "3DES")
};

static std::map<gdcm::CryptographicMessageSyntax::CipherTypes, std::string> cip2str(cip2str_data,
    cip2str_data + sizeof cip2str_data / sizeof cip2str_data[0]);

const char * const tstr = "12345";
const size_t tstr_l = strlen(tstr);
#define BUFSZ 5000

bool TestCMSProvider(gdcm::CryptographicMessageSyntax& cms, const char * provName)
{
  const std::string certpath = gdcm::Filename::Join(gdcm::Testing::GetSourceDirectory(), "/Testing/Source/Data/certificate.pem" );
  const std::string keypath = gdcm::Filename::Join(gdcm::Testing::GetSourceDirectory(), "/Testing/Source/Data/privatekey.pem" );
  const std::string encrypted_vector = gdcm::Filename::Join(gdcm::Testing::GetSourceDirectory(), "/Testing/Source/Data/encrypted_text" );

  bool ret = true;
  for (unsigned int i = 0; i < 4; i++)
    {
    char encout[BUFSZ] = {0}, decout[BUFSZ] = {0};
    size_t encoutlen = BUFSZ, decoutlen = BUFSZ;
    cms.SetCipherType(ciphers[i]);
    bool encryptSuccess = cms.Encrypt(encout, encoutlen, tstr, tstr_l);
    if (!encryptSuccess)
      {
      std::cerr << provName << " using " << cip2str[ciphers[i]] << ": encryption failed" << std::endl;
      ret = false;
      continue;
      }
    bool decryptSuccess = cms.Decrypt(decout, decoutlen, encout, encoutlen);
    if (!decryptSuccess)
      {
      std::cerr << provName << " using " << cip2str[ciphers[i]] << ": decryption failed" << std::endl;
      ret = false;
      continue;
      }
    if (decoutlen != tstr_l)
      {
      std::cerr << provName << " using " << cip2str[ciphers[i]] << ": decryted length different from original (" << decoutlen << " != " << tstr_l << ")" << std::endl;
      ret = false;
      continue;
      }
    if (memcmp(tstr, decout, tstr_l) != 0)
      {
      std::cerr << provName << " using " << cip2str[ciphers[i]] << ": decryted data different from original" << std::endl;
      ret = false;
      continue;
      }
    }
  
  return ret;
}

bool TestCMSVector(gdcm::CryptographicMessageSyntax& cms, const char * provName)
{
  char decout[BUFSZ] = {0};
  size_t decoutlen = BUFSZ;
  std::string encrypted_filename = gdcm::Filename::Join(gdcm::Testing::GetSourceDirectory(), "/Testing/Source/Data/encrypted_text" );
  const char * tv_plaintext = "1234567890abcdefghijklmnopqrstuvwxyz";
  size_t tv_plaintext_len = strlen(tv_plaintext);

  char * test_vector;
  // FIXME : should I delete test_vector ?
  size_t tvlen;
  if (!LoadFile(encrypted_filename.c_str(), test_vector, tvlen))
    {
    std::cerr << "Couldn't load encrypted file: " << encrypted_filename << std::endl;
    return false;
    }
  bool decryptSuccess = cms.Decrypt(decout, decoutlen, test_vector, tvlen);
  if (!decryptSuccess)
    {
    std::cerr << provName << " test vector decryption failed" << std::endl;
    return false;
    }
  if (decoutlen != tv_plaintext_len)
    {
    std::cerr << provName << " test vector decryted length different from original (" << decoutlen << " != " << tstr_l << ")" << std::endl;
    return false;
    }
  if (memcmp(tv_plaintext, decout, tv_plaintext_len) != 0)
    {
    std::cerr << provName << " test vector decryted data different from original" << std::endl;
    return false;
    }

  return true;
}

bool TestCMSCompatibility(gdcm::CryptographicMessageSyntax& cms1, const char * provName1, gdcm::CryptographicMessageSyntax& cms2, const char * provName2)
{
  const std::string encrypted_vector = gdcm::Filename::Join(gdcm::Testing::GetSourceDirectory(), "/Testing/Source/Data/encrypted_text" );

  bool ret = true;
  for (int i = 0; i < 4; i++)
    {
    char encout[BUFSZ] = {0}, decout[BUFSZ] = {0};
    size_t encoutlen = BUFSZ, decoutlen = BUFSZ;
    cms1.SetCipherType(ciphers[i]);
    cms2.SetCipherType(ciphers[i]);

    bool encryptSuccess = cms1.Encrypt(encout, encoutlen, tstr, tstr_l);
    if (!encryptSuccess)
      {
      std::cerr << provName1 << " & " << provName2 << " using " << cip2str[ciphers[i]] << ": encryption failed" << std::endl;
      ret = false;
      break;
      }
    bool decryptSuccess = cms2.Decrypt(decout, decoutlen, encout, encoutlen);
    if (!decryptSuccess)
      {
      std::cerr << provName1 << " & " << provName2 << " using " << cip2str[ciphers[i]] << ": decryption failed" << std::endl;
      ret = false;
      break;
      }
    if (decoutlen != tstr_l)
      {
      std::cerr <<  provName1 << " & " << provName2 << " using " << cip2str[ciphers[i]] << ": decryted length different from original (" << decoutlen << " != " << tstr_l << ")" << std::endl;
      ret = false;
      break;
      }
    if (memcmp(tstr, decout, tstr_l) != 0)
      {
      std::cerr <<  provName1 << " & " << provName2 << " using " << cip2str[ciphers[i]] << ": decryted data different from original" << std::endl;
      ret = false;
      break;
      }
    }

  /*
    char encout[BUFSZ] = {0}, decout[BUFSZ] = {0};
    size_t encoutlen = BUFSZ, decoutlen = BUFSZ;
    for (int i = 0; i < 4; i++)
    {
    bool ret = true;
    char encout[BUFSZ] = {0}, decout[BUFSZ] = {0};
    size_t encoutlen = BUFSZ, decoutlen = BUFSZ;
    cms1.SetCipherType(ciphers[i]);
    cms2.SetCipherType(ciphers[i]);
    //cms2.Encrypt(encout, encoutlen, tstr, tstr_l);
    //cms1.Decrypt(decout, decoutlen, encout, encoutlen);
    //assert(decoutlen == tstr_l);
    //assert(memcmp(tstr, decout, tstr_l) == 0);

    bool encryptSuccess = cms1.Encrypt(encout, encoutlen, tstr, tstr_l);
    if (!encryptSuccess)
      {
      std::cerr << provName1 << " & " << provName2 << " using " << cip2str[ciphers[i]] << ": encryption failed" << std::endl;
      ret = false;
      break;
      }
    bool decryptSuccess = cms2.Decrypt(decout, decoutlen, encout, encoutlen);
    if (!decryptSuccess)
      {
      std::cerr << provName1 << " & " << provName2 << " using " << cip2str[ciphers[i]] << ": decryption failed" << std::endl;
      ret = false;
      break;
      }
    if (decoutlen != tstr_l)
      {
      std::cerr <<  provName1 << " & " << provName2 << " using " << cip2str[ciphers[i]] << ": decryted length different from original (" << decoutlen << " != " << tstr_l << ")" << std::endl;
      ret = false;
      break;
      }
    if (memcmp(tstr, decout, tstr_l) != 0)
      {
      std::cerr <<  provName1 << " & " << provName2 << " using " << cip2str[ciphers[i]] << ": decryted data different from original" << std::endl;
      ret = false;
      break;
      }
    }*/
  return ret;
}

int TestCryptographicMessageSyntax(int, char *[])
{
  std::string certpath = gdcm::Filename::Join(gdcm::Testing::GetSourceDirectory(), "/Testing/Source/Data/certificate.pem" );
  std::string keypath = gdcm::Filename::Join(gdcm::Testing::GetSourceDirectory(), "/Testing/Source/Data/privatekey.pem" );
  bool bret = true;
  //typedef std::tuple<std::string, gdcm::CryptographicMessageSyntax&> StringCMSPairType;
  std::vector<gdcm::CryptographicMessageSyntax*> availableCMS;
  std::vector<std::string> availableCMSName;


#ifdef GDCM_USE_SYSTEM_OPENSSL
  gdcm::CryptoFactory* osslp7 = gdcm::CryptoFactory::GetFactoryInstance(gdcm::CryptoFactory::OPENSSLP7);
  std::auto_ptr<gdcm::CryptographicMessageSyntax> ocmsp7(osslp7->CreateCMSProvider());
  ocmsp7->ParseKeyFile(keypath.c_str());
  ocmsp7->ParseCertificateFile(certpath.c_str());
  bret = TestCMSProvider(*ocmsp7, "OpenSSL PKCS7") && bret;
  bret = TestCMSVector(*ocmsp7, "OpenSSL PKCS7") && bret;
  availableCMS.push_back(ocmsp7.get());
  availableCMSName.push_back("OpenSSL PKCS7");
#endif

#ifdef GDCM_USE_SYSTEM_OPENSSL
#ifdef GDCM_HAVE_CMS_RECIPIENT_PASSWORD
  gdcm::CryptoFactory* ossl = gdcm::CryptoFactory::GetFactoryInstance(gdcm::CryptoFactory::OPENSSL);
  std::auto_ptr<gdcm::CryptographicMessageSyntax> ocms(ossl->CreateCMSProvider());
  ocms->ParseKeyFile(keypath.c_str());
  ocms->ParseCertificateFile(certpath.c_str());
  bret = TestCMSProvider(*ocms, "OpenSSL CMS") && bret;
  bret = TestCMSVector(*ocms, "OpenSSL CMS") && bret;
  availableCMS.push_back(ocms.get());
  availableCMSName.push_back("OpenSSL CMS");
#endif
#endif

#ifdef WIN32
  gdcm::CryptoFactory* capi = gdcm::CryptoFactory::GetFactoryInstance(gdcm::CryptoFactory::CAPI);
  std::auto_ptr<gdcm::CryptographicMessageSyntax> ccms(capi->CreateCMSProvider());
  ccms->ParseCertificateFile(certpath.c_str());
  ccms->ParseKeyFile(keypath.c_str());
  bret = TestCMSProvider(*ccms, "CAPI") && bret;
  bret = TestCMSVector(*ccms, "CAPI") && bret;
  availableCMS.push_back(ccms.get());
  availableCMSName.push_back("CAPI");
#endif

  for (size_t i = 0; i < availableCMS.size(); ++i)
    for (size_t j = i+1; j < availableCMS.size(); ++j)
      bret = TestCMSCompatibility(*availableCMS[i], availableCMSName[i].c_str(), *availableCMS[j], availableCMSName[j].c_str()) && bret;

  return (bret ? 0 : 1);
}

int TestPasswordBasedEncryption(int, char *[])
{
  const char *directory = gdcm::Testing::GetDataRoot();
  std::string encrypted_dicomdir =
    gdcm::Filename::Join(directory, "/securedicomfileset/DICOMDIR" );
  std::string encrypted_image =
    gdcm::Filename::Join(directory, "/securedicomfileset/IMAGES/IMAGE1" );

#ifdef GDCM_USE_SYSTEM_OPENSSL
  gdcm::CryptoFactory* ossl = gdcm::CryptoFactory::GetFactoryInstance(gdcm::CryptoFactory::OPENSSL);
  std::auto_ptr<gdcm::CryptographicMessageSyntax> ocms(ossl->CreateCMSProvider());

  ocms->SetPassword("password", strlen("password"));
  if (!TestCMSProvider(*ocms, "OpenSSL"))
    return 1;
  
  char decout[BUFSZ] = {0};
  size_t decoutlen = BUFSZ;
  char * ddir = new char[5000];
  size_t ddirlen = 5000;
  LoadFile(encrypted_dicomdir.c_str(), ddir, ddirlen);
  bool decryptSuccess = ocms->Decrypt(decout, decoutlen, ddir, ddirlen);
  if (!decryptSuccess)
    {
    std::cerr << "OpenSSL sample DICOMDIR decryption failed" << std::endl;
    return 1;
    }
  if (decoutlen == 0)
    {
    std::cerr << "OpenSSL sample DICOMDIR decrypted length == 0" << std::endl;
    return 1;
    }
  
#endif

  return 0;
}
