#ifndef _CONLL_READER_
#define _CONLL_READER_

#include "Reader.h"
#include "MyLib.h"
#include "Utf.h"
#include <sstream>

using namespace std;
/*
 this class reads conll-format data (10 columns, no srl-info)
 */
class InstanceReader: public Reader {
public:
  InstanceReader() {
  }
  ~InstanceReader() {
  }

  Instance *getNext() {
    m_instance.clear();

    vector<string> vecLine;
    while (1) {
      string strLine;
      if (!my_getline(m_inf, strLine)) {
        break;
      }
      if (strLine.empty())
        break;
      vecLine.push_back(strLine);
    }

    int length = vecLine.size();

    if (length == 1) {
      m_instance.allocate(1);
      vector<string> vecInfo;
      split_bychar(vecLine[0], vecInfo, ' ');
      int veclength = vecInfo.size();
      m_instance.label = vecInfo[0];
      for (int j = 1; j < veclength; j++) {
        m_instance.words[0].push_back(vecInfo[j]);
        vector<string> curChars;
        getCharactersFromUTF8String(vecInfo[j], curChars);
        m_instance.chars[0].push_back(curChars);
      }
    } else {
      m_instance.allocate(2);
      for (int i = 0; i < length; ++i) {
        vector<string> vecInfo;
        split_bychar(vecLine[i], vecInfo, ' ');
        int veclength = vecInfo.size();
        if (i == length - 1) {
          m_instance.label = vecInfo[0];
          for (int j = 1; j < veclength; j++) {
            m_instance.words[1].push_back(vecInfo[j]);
            vector<string> curChars;
            getCharactersFromUTF8String(vecInfo[j], curChars);
            m_instance.chars[1].push_back(curChars);
          }
        } else {
          for (int j = 1; j < veclength; j++) {
            m_instance.words[0].push_back(vecInfo[j]);
            vector<string> curChars;
            getCharactersFromUTF8String(vecInfo[j], curChars);
            m_instance.chars[0].push_back(curChars);
          }
        }

      }
    }

    return &m_instance;
  }
};

#endif

