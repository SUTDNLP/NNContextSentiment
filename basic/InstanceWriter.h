#ifndef _CONLL_WRITER_
#define _CONLL_WRITER_

#include "Writer.h"
#include <sstream>

using namespace std;
/*
	this class writes conll-format result (no srl-info).
*/
class InstanceWriter : public Writer
{
public:
	InstanceWriter(){}
	~InstanceWriter(){}
	int write(const Instance *pInstance)
	{
	  if (!m_outf.is_open())
	    return -1;

	  const vector<vector<string> > &words = pInstance->words;

	  for (int i = 0; i < words.size(); ++i) {
	    if (i < words.size() - 1)
	      m_outf << "history" << endl;
	    else if(pInstance->confidence < 0.0)
	      m_outf << pInstance->label << endl;
	    else
	      m_outf << pInstance->label << " " << pInstance->confidence << endl;
	    //for (int j = 0; j < words[i].size(); j++) {
	    //  m_outf << " " << words[i][j];
	    //}
	  }
	  m_outf << endl;
	  return 0;
	}
};

#endif

