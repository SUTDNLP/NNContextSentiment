/*
 * Labeler.cpp
 *
 *  Created on: Mar 16, 2015
 *      Author: mszhang
 */

#include "SparseDetector.h"

#include "Argument_helper.h"

Labeler::Labeler() {
  // TODO Auto-generated constructor stub
  nullkey = "-null-";
  unknownkey = "-unknown-";
  seperateKey = "#";

}

Labeler::~Labeler() {
  // TODO Auto-generated destructor stub
}

int Labeler::createAlphabet(const vector<Instance>& vecInsts) {
  cout << "Creating Alphabet..." << endl;

  int numInstance, labelId;
  hash_map<string, int> word_stat;
  hash_map<string, int> char_stat;
  m_labelAlphabet.clear();

  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance *pInstance = &vecInsts[numInstance];

    const vector<vector<string> > &words = pInstance->words;
    const vector<vector<vector<string> > > &chars = pInstance->chars;

    const string &label = pInstance->label;
    labelId = m_labelAlphabet.from_string(label);

    int curInstSize = words.size();
    for (int i = 0; i < curInstSize; ++i) {
      int curWordSize = words[i].size();
      for (int j = 0; j < curWordSize; j++) {
        string curword = normalize_to_lowerwithdigit(words[i][j]);
        word_stat[curword]++;
        int curWordLength = chars[i][j].size();
        for (int k = 0; k < curWordLength; k++)
          char_stat[chars[i][j][k]]++;
      }
    }

    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }

  cout << numInstance << " " << endl;
  cout << "Label num: " << m_labelAlphabet.size() << endl;
  cout << "Total word num: " << word_stat.size() << endl;
  cout << "Total char num: " << char_stat.size() << endl;

  m_wordAlphabet.clear();
  m_wordAlphabet.from_string(nullkey);
  m_wordAlphabet.from_string(unknownkey);
  m_charAlphabet.clear();
  m_charAlphabet.from_string(nullkey);
  m_charAlphabet.from_string(unknownkey);

  hash_map<string, int>::iterator feat_iter;

  for (feat_iter = word_stat.begin(); feat_iter != word_stat.end(); feat_iter++) {
    if (!m_options.wordEmbFineTune || feat_iter->second > m_options.wordCutOff) {
      m_wordAlphabet.from_string(feat_iter->first);
    }
  }

  for (feat_iter = char_stat.begin(); feat_iter != char_stat.end(); feat_iter++) {
    if (!m_options.charEmbFineTune || feat_iter->second > m_options.charCutOff) {
      m_charAlphabet.from_string(feat_iter->first);
      //std::cout << feat_iter->first << std::endl;
    }
  }

  cout << "Remain words num: " << m_wordAlphabet.size() << endl;
  cout << "Remain char num: " << m_charAlphabet.size() << endl;

  m_labelAlphabet.set_fixed_flag(true);
  m_wordAlphabet.set_fixed_flag(true);
  m_charAlphabet.set_fixed_flag(true);

  if (m_linearfeat > 0) {
    cout << "Extracting linear features...." << endl;
    hash_map<string, int> feature_stat;
    for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
      const Instance *pInstance = &vecInsts[numInstance];
      vector<string> features;
      extractLinearFeatures(features, pInstance);
      for (int j = 0; j < features.size(); j++)
        feature_stat[features[j]]++;
    }

    cout << numInstance << " " << endl;
    cout << "Total feature num: " << feature_stat.size() << endl;
    m_featAlphabet.clear();

    for (feat_iter = feature_stat.begin(); feat_iter != feature_stat.end(); feat_iter++) {
      if (feat_iter->second > m_options.featCutOff) {
        m_featAlphabet.from_string(feat_iter->first);
      }
    }
    cout << "Remain feature num: " << m_featAlphabet.size() << endl;
    m_featAlphabet.set_fixed_flag(true);
  }

  return 0;
}

int Labeler::addTestWordAlpha(const vector<Instance>& vecInsts) {
  cout << "Adding word Alphabet..." << endl;

  int numInstance;
  hash_map<string, int> word_stat;
  m_wordAlphabet.set_fixed_flag(false);

  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance *pInstance = &vecInsts[numInstance];

    const vector<vector<string> > &words = pInstance->words;

    int curInstSize = words.size();
    for (int i = 0; i < curInstSize; ++i) {
      int curWordSize = words[i].size();
      for (int j = 0; j < curWordSize; j++) {
        string curword = normalize_to_lowerwithdigit(words[i][j]);
        word_stat[curword]++;
      }
    }

    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }

  hash_map<string, int>::iterator feat_iter;
  for (feat_iter = word_stat.begin(); feat_iter != word_stat.end(); feat_iter++) {
    if (!m_options.wordEmbFineTune || feat_iter->second > m_options.wordCutOff) {
      m_wordAlphabet.from_string(feat_iter->first);
    }
  }

  m_wordAlphabet.set_fixed_flag(true);

  return 0;
}

int Labeler::addTestCharAlpha(const vector<Instance>& vecInsts) {
  cout << "Adding char Alphabet..." << endl;

  int numInstance;
  hash_map<string, int> char_stat;
  m_charAlphabet.set_fixed_flag(false);

  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance *pInstance = &vecInsts[numInstance];

    const vector<vector<vector<string> > > &chars = pInstance->chars;

    int curInstSize = chars.size();
    for (int i = 0; i < curInstSize; ++i) {
      int curWordSize = chars[i].size();
      for (int j = 0; j < curWordSize; j++) {
        int curWordLength = chars[i][j].size();
        for (int k = 0; k < curWordLength; k++)
          char_stat[chars[i][j][k]]++;
      }
    }

    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }

  hash_map<string, int>::iterator feat_iter;
  for (feat_iter = char_stat.begin(); feat_iter != char_stat.end(); feat_iter++) {
    if (!m_options.charEmbFineTune || feat_iter->second > m_options.charCutOff) {
      m_charAlphabet.from_string(feat_iter->first);
    }
  }

  m_charAlphabet.set_fixed_flag(true);

  return 0;
}

void Labeler::extractLinearFeatures(vector<string>& features, const Instance* pInstance) {
  features.clear();
  const vector<vector<string> >& words = pInstance->words;
  int seq_size = words.size();
  if (seq_size > 2) {
    cout << "error input, two or more histories..." << endl;
  }

  string feat = "";
  const vector<string> lastWords = words[seq_size - 1];
  int wordnumber = lastWords.size();
  for (int i = 0; i < wordnumber; i++) {
    feat = "F1U=" + lastWords[i];
    features.push_back(feat);
    string prevword = i - 1 >= 0 ? lastWords[i - 1] : nullkey;
    feat = "F2B=" + prevword + seperateKey + lastWords[i];
    features.push_back(feat);
//    string prev2word = i - 2 >= 0 ? lastWords[i - 2] : nullkey;
//    feat = "F3T=" + prev2word + seperateKey + prevword + seperateKey + lastWords[i];
//    features.push_back(feat);
  }

  if (m_linearfeat > 1 && seq_size == 2) {
    vector<string> lastWords = words[seq_size - 2];
    wordnumber = lastWords.size();
    for (int i = 0; i < wordnumber; i++) {
      feat = "F4U=" + lastWords[i];
      features.push_back(feat);
    }
  }

}

void Labeler::extractFeature(Feature& feat, const Instance* pInstance, int idx) {
  feat.clear();

  const vector<vector<string> >& words = pInstance->words;
  const vector<vector<vector<string> > > &chars = pInstance->chars;

  static vector<int> curChars;

  int sentsize = words.size();

  if (idx < 0 || idx >= sentsize)
    return;

  int wordnumber = words[idx].size();

  int unknownWordId = m_wordAlphabet.from_string(unknownkey);
  int unknownCharId = m_charAlphabet.from_string(unknownkey);

  for (int i = 0; i < wordnumber; i++) {
    string curWord = normalize_to_lowerwithdigit(words[idx][i]);
    int curWordId = m_wordAlphabet.from_string(curWord);
    if (curWordId >= 0)
      feat.words.push_back(curWordId);
    else
      feat.words.push_back(unknownWordId);

    int wordlength = chars[idx][i].size();
    curChars.clear();
    for (int j = 0; j < wordlength; j++) {
      string curChar = chars[idx][i][j];
      int curCharId = m_charAlphabet.from_string(curChar);
      if (curCharId >= 0)
        curChars.push_back(curCharId);
      else
        curChars.push_back(unknownCharId);
    }
    feat.chars.push_back(curChars);
  }
}

void Labeler::convert2Example(const Instance* pInstance, Example& exam) {
  exam.clear();
  const string &label = pInstance->label;
  const vector<vector<string> > &words = pInstance->words;
  const vector<vector<vector<string> > > &chars = pInstance->chars;

  int numLabels = m_labelAlphabet.size();
  for (int j = 0; j < numLabels; ++j) {
    string str = m_labelAlphabet.from_id(j);
    if (str.compare(label) == 0)
      exam.m_labels.push_back(1);
    else
      exam.m_labels.push_back(0);
  }

  int curInstSize = words.size();
  for (int i = 0; i < curInstSize; ++i) {
    Feature feat;
    extractFeature(feat, pInstance, i);
    exam.m_features.push_back(feat);
  }
  if (m_linearfeat > 0) {
    vector<string> linear_features;
    extractLinearFeatures(linear_features, pInstance);
    for (int i = 0; i < linear_features.size(); i++) {
      int curFeatId = m_featAlphabet.from_string(linear_features[i]);
      if (curFeatId >= 0)
        exam.m_linears.push_back(curFeatId);
    }
  }
}

void Labeler::initialExamples(const vector<Instance>& vecInsts, vector<Example>& vecExams) {
  int numInstance;
  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance *pInstance = &vecInsts[numInstance];
    Example curExam;
    convert2Example(pInstance, curExam);
    vecExams.push_back(curExam);

    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }

  cout << numInstance << " " << endl;
}

void Labeler::train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile,
    const string& wordEmbFile, const string& charEmbFile) {
  if (optionFile != "")
    m_options.load(optionFile);

  m_options.showOptions();
  m_linearfeat = m_options.linearfeatCat;
  if(m_linearfeat <= 0)
  {
    m_linearfeat = 2;
  }


  vector<Instance> trainInsts, devInsts, testInsts;
  static vector<Instance> decodeInstResults;
  static Instance curDecodeInst;
  bool bCurIterBetter = false;

  m_pipe.readInstances(trainFile, trainInsts, m_options.maxInstance);
  if (devFile != "")
    m_pipe.readInstances(devFile, devInsts, m_options.maxInstance);
  if (testFile != "")
    m_pipe.readInstances(testFile, testInsts, m_options.maxInstance);

  //Ensure that each file in m_options.testFiles exists!
  vector<vector<Instance> > otherInsts(m_options.testFiles.size());
  for (int idx = 0; idx < m_options.testFiles.size(); idx++) {
    m_pipe.readInstances(m_options.testFiles[idx], otherInsts[idx], m_options.maxInstance);
  }

  //std::cout << "Training example number: " << trainInsts.size() << std::endl;
  //std::cout << "Dev example number: " << trainInsts.size() << std::endl;
  //std::cout << "Test example number: " << trainInsts.size() << std::endl;

  createAlphabet(trainInsts);

  if (!m_options.wordEmbFineTune) {
    addTestWordAlpha(devInsts);
    addTestWordAlpha(testInsts);
    for (int idx = 0; idx < otherInsts.size(); idx++) {
      addTestWordAlpha(otherInsts[idx]);
    }
    cout << "Remain words num: " << m_wordAlphabet.size() << endl;
  }

  if (!m_options.charEmbFineTune) {
    addTestCharAlpha(devInsts);
    addTestCharAlpha(testInsts);
    for (int idx = 0; idx < otherInsts.size(); idx++) {
      addTestCharAlpha(otherInsts[idx]);
    }
    cout << "Remain char num: " << m_charAlphabet.size() << endl;
  }

  NRMat<dtype> wordEmb;
  if (wordEmbFile != "") {
    readWordEmbeddings(wordEmbFile, wordEmb);
  } else {
    wordEmb.resize(m_wordAlphabet.size(), m_options.wordEmbSize);
    wordEmb.randu(1000);
  }

  NRMat<dtype> charEmb;
  if (charEmbFile != "") {
    readWordEmbeddings(charEmbFile, charEmb);
  } else {
    charEmb.resize(m_charAlphabet.size(), m_options.charEmbSize);
    charEmb.randu(1001);
  }

  m_classifier.init(m_labelAlphabet.size(), m_featAlphabet.size());
  m_classifier.setDropValue(m_options.dropProb);


  vector<Example> trainExamples, devExamples, testExamples;
  initialExamples(trainInsts, trainExamples);
  initialExamples(devInsts, devExamples);
  initialExamples(testInsts, testExamples);

  vector<int> otherInstNums(otherInsts.size());
  vector<vector<Example> > otherExamples(otherInsts.size());
  for (int idx = 0; idx < otherInsts.size(); idx++) {
    initialExamples(otherInsts[idx], otherExamples[idx]);
    otherInstNums[idx] = otherExamples[idx].size();
  }

  dtype bestDIS = 0;

  int inputSize = trainExamples.size();

  srand(0);
  std::vector<int> indexes;
  for (int i = 0; i < inputSize; ++i)
    indexes.push_back(i);

  static Metric eval, metric_dev, metric_test;
  static vector<Example> subExamples;
  int devNum = devExamples.size(), testNum = testExamples.size();

  int maxIter = m_options.maxIter;
  if (m_options.batchSize > 1)
    maxIter = m_options.maxIter * (inputSize / m_options.batchSize + 1);

  dtype cost = 0.0;
  std::cout << "maxIter = " << maxIter << std::endl;
  for (int iter = 0; iter < m_options.maxIter; ++iter) {
    std::cout << "##### Iteration " << iter << std::endl;
    eval.reset();
    if (m_options.batchSize == 1) {
      random_shuffle(indexes.begin(), indexes.end());
      for (int updateIter = 0; updateIter < inputSize; updateIter++) {
        subExamples.clear();
        int start_pos = updateIter;
        int end_pos = (updateIter + 1);
        if (end_pos > inputSize)
          end_pos = inputSize;

        for (int idy = start_pos; idy < end_pos; idy++) {
          subExamples.push_back(trainExamples[indexes[idy]]);
        }

        int curUpdateIter = iter * inputSize + updateIter;
        cost = m_classifier.process(subExamples, curUpdateIter);

        eval.overall_label_count += m_classifier._eval.overall_label_count;
        eval.correct_label_count += m_classifier._eval.correct_label_count;

        if ((curUpdateIter + 1) % m_options.verboseIter == 0) {
          //m_classifier.checkgrads(subExamples, curUpdateIter+1);
          std::cout << "current: " << updateIter + 1 << ", total instances: " << inputSize << std::endl;
          std::cout << "Cost = " << cost << ", SA Correct(%) = " << eval.getAccuracy() << std::endl;
        }
        m_classifier.updateParams(m_options.regParameter, m_options.adaAlpha, m_options.adaEps);
      }
    } else {
      cost = 0.0;
      for (int updateIter = 0; updateIter < m_options.verboseIter; updateIter++) {
        random_shuffle(indexes.begin(), indexes.end());
        subExamples.clear();
        for (int idy = 0; idy < m_options.batchSize; idy++) {
          subExamples.push_back(trainExamples[indexes[idy]]);
        }
        int curUpdateIter = iter * m_options.verboseIter + updateIter;
        cost += m_classifier.process(subExamples, curUpdateIter);
       //m_classifier.checkgrads(subExamples, curUpdateIter);
        eval.overall_label_count += m_classifier._eval.overall_label_count;
        eval.correct_label_count += m_classifier._eval.correct_label_count;

        m_classifier.updateParams(m_options.regParameter, m_options.adaAlpha, m_options.adaEps);
      }
      std::cout << "current iter: " << iter + 1 << ", total iter: " << maxIter << std::endl;
      std::cout << "Cost = " << cost << ", SA Correct(%) = " << eval.getAccuracy() << std::endl;
    }

    if (devNum > 0) {
      bCurIterBetter = false;
      if (!m_options.outBest.empty())
        decodeInstResults.clear();
      metric_dev.reset();
      for (int idx = 0; idx < devExamples.size(); idx++) {
        string result_label;
        dtype confidence = predict(devExamples[idx].m_linears, devExamples[idx].m_features, result_label);

        devInsts[idx].Evaluate(result_label, metric_dev);

        if (!m_options.outBest.empty()) {
          curDecodeInst.copyValuesFrom(devInsts[idx]);
          curDecodeInst.assignLabel(result_label, confidence);
          decodeInstResults.push_back(curDecodeInst);
        }
      }
      metric_dev.print();

      if ((!m_options.outBest.empty() && metric_dev.getAccuracy() > bestDIS)) {
        m_pipe.outputAllInstances(devFile + m_options.outBest, decodeInstResults);
        bCurIterBetter = true;
      }

      if (testNum > 0) {
        if (!m_options.outBest.empty())
          decodeInstResults.clear();
        metric_test.reset();
        for (int idx = 0; idx < testExamples.size(); idx++) {
          string result_label;
          dtype confidence = predict(testExamples[idx].m_linears, testExamples[idx].m_features, result_label);
          testInsts[idx].Evaluate(result_label, metric_test);

          if (bCurIterBetter && !m_options.outBest.empty()) {
            curDecodeInst.copyValuesFrom(testInsts[idx]);
            curDecodeInst.assignLabel(result_label, confidence);
            decodeInstResults.push_back(curDecodeInst);
          }
        }
        std::cout << "test:" << std::endl;
        metric_test.print();

        if ((!m_options.outBest.empty() && bCurIterBetter)) {
          m_pipe.outputAllInstances(testFile + m_options.outBest, decodeInstResults);
        }
      }

      for (int idx = 0; idx < otherExamples.size(); idx++) {
        std::cout << "processing " << m_options.testFiles[idx] << std::endl;
        if (!m_options.outBest.empty())
          decodeInstResults.clear();
        metric_test.reset();
        for (int idy = 0; idy < otherExamples[idx].size(); idy++) {
          string result_label;
          dtype confidence = predict(otherExamples[idx][idy].m_linears, otherExamples[idx][idy].m_features, result_label);

          otherInsts[idx][idy].Evaluate(result_label, metric_test);

          if (bCurIterBetter && !m_options.outBest.empty()) {
            curDecodeInst.copyValuesFrom(otherInsts[idx][idy]);
            curDecodeInst.assignLabel(result_label, confidence);
            decodeInstResults.push_back(curDecodeInst);
          }
        }
        std::cout << "test:" << std::endl;
        metric_test.print();

        if ((!m_options.outBest.empty() && bCurIterBetter)) {
          m_pipe.outputAllInstances(m_options.testFiles[idx] + m_options.outBest, decodeInstResults);
        }
      }

      if ((m_options.saveIntermediate && metric_dev.getAccuracy() > bestDIS)) {
        if (metric_dev.getAccuracy() > bestDIS) {
          std::cout << "Exceeds best previous performance of " << bestDIS << ". Saving model file.." << std::endl;
          bestDIS = metric_dev.getAccuracy();
        }
        writeModelFile(modelFile);
      }

    }
    // Clear gradients
  }

  if (devNum > 0) {
    bCurIterBetter = false;
    if (!m_options.outBest.empty())
      decodeInstResults.clear();
    metric_dev.reset();
    for (int idx = 0; idx < devExamples.size(); idx++) {
      string result_label;
      dtype confidence = predict(devExamples[idx].m_linears, devExamples[idx].m_features, result_label);

      devInsts[idx].Evaluate(result_label, metric_dev);

      if (!m_options.outBest.empty()) {
        curDecodeInst.copyValuesFrom(devInsts[idx]);
        curDecodeInst.assignLabel(result_label, confidence);
        decodeInstResults.push_back(curDecodeInst);
      }
    }
    metric_dev.print();

    if ((!m_options.outBest.empty() && metric_dev.getAccuracy() > bestDIS)) {
      m_pipe.outputAllInstances(devFile + m_options.outBest, decodeInstResults);
      bCurIterBetter = true;
    }

    if (testNum > 0) {
      if (!m_options.outBest.empty())
        decodeInstResults.clear();
      metric_test.reset();
      for (int idx = 0; idx < testExamples.size(); idx++) {
        string result_label;
        dtype confidence = predict(testExamples[idx].m_linears, testExamples[idx].m_features, result_label);
        testInsts[idx].Evaluate(result_label, metric_test);

        if (bCurIterBetter && !m_options.outBest.empty()) {
          curDecodeInst.copyValuesFrom(testInsts[idx]);
          curDecodeInst.assignLabel(result_label, confidence);
          decodeInstResults.push_back(curDecodeInst);
        }
      }
      std::cout << "test:" << std::endl;
      metric_test.print();

      if ((!m_options.outBest.empty() && bCurIterBetter)) {
        m_pipe.outputAllInstances(testFile + m_options.outBest, decodeInstResults);
      }
    }

    for (int idx = 0; idx < otherExamples.size(); idx++) {
      std::cout << "processing " << m_options.testFiles[idx] << std::endl;
      if (!m_options.outBest.empty())
        decodeInstResults.clear();
      metric_test.reset();
      for (int idy = 0; idy < otherExamples[idx].size(); idy++) {
        string result_label;
        dtype confidence = predict(otherExamples[idx][idy].m_linears, otherExamples[idx][idy].m_features, result_label);

        otherInsts[idx][idy].Evaluate(result_label, metric_test);

        if (bCurIterBetter && !m_options.outBest.empty()) {
          curDecodeInst.copyValuesFrom(otherInsts[idx][idy]);
          curDecodeInst.assignLabel(result_label, confidence);
          decodeInstResults.push_back(curDecodeInst);
        }
      }
      std::cout << "test:" << std::endl;
      metric_test.print();

      if ((!m_options.outBest.empty() && bCurIterBetter)) {
        m_pipe.outputAllInstances(m_options.testFiles[idx] + m_options.outBest, decodeInstResults);
      }
    }

    if ((m_options.saveIntermediate && metric_dev.getAccuracy() > bestDIS)) {
      if (metric_dev.getAccuracy() > bestDIS) {
        std::cout << "Exceeds best previous performance of " << bestDIS << ". Saving model file.." << std::endl;
        bestDIS = metric_dev.getAccuracy();
      }
      writeModelFile(modelFile);
    }

  } else {
    writeModelFile(modelFile);
  }
}

dtype Labeler::predict(const vector<int>& linears, const vector<Feature>& features, string& output) {
  vector<dtype> labelprobs;
  int label = m_classifier.predict(linears, features, labelprobs);
  output = m_labelAlphabet.from_id(label);
  return labelprobs[label];
}

void Labeler::test(const string& testFile, const string& outputFile, const string& modelFile) {
  loadModelFile(modelFile);
  vector<Instance> testInsts;
  m_pipe.readInstances(testFile, testInsts);

  vector<Example> testExamples;
  initialExamples(testInsts, testExamples);

  int testNum = testExamples.size();
  vector<Instance> testInstResults;
  Metric metric_test;
  metric_test.reset();
  for (int idx = 0; idx < testExamples.size(); idx++) {
    string result_label;
    dtype confidence = predict(testExamples[idx].m_linears, testExamples[idx].m_features, result_label);
    testInsts[idx].Evaluate(result_label, metric_test);
    Instance curResultInst;
    curResultInst.copyValuesFrom(testInsts[idx]);
    curResultInst.assignLabel(result_label, confidence);
    testInstResults.push_back(curResultInst);
  }
  std::cout << "test:" << std::endl;
  metric_test.print();

  m_pipe.outputAllInstances(outputFile, testInstResults);

}

void Labeler::readWordEmbeddings(const string& inFile, NRMat<dtype>& wordEmb) {
  static ifstream inf;
  if (inf.is_open()) {
    inf.close();
    inf.clear();
  }
  inf.open(inFile.c_str());

  static string strLine, curWord;
  static int wordId;

  //find the first line, decide the wordDim;
  while (1) {
    if (!my_getline(inf, strLine)) {
      break;
    }
    if (!strLine.empty())
      break;
  }

  int unknownId = m_wordAlphabet.from_string(unknownkey);

  static vector<string> vecInfo;
  split_bychar(strLine, vecInfo, ' ');
  int wordDim = vecInfo.size() - 1;

  std::cout << "word embedding dim is " << wordDim << std::endl;
  m_options.wordEmbSize = wordDim;

  wordEmb.resize(m_wordAlphabet.size(), wordDim);
  wordEmb = 0.0;
  curWord = normalize_to_lowerwithdigit(vecInfo[0]);
  wordId = m_wordAlphabet.from_string(curWord);
  hash_set<int> indexers;
  dtype sum[wordDim];
  int count = 0;
  bool bHasUnknown = false;
  if (wordId >= 0) {
    count++;
    if (unknownId == wordId)
      bHasUnknown = true;
    indexers.insert(wordId);
    for (int idx = 0; idx < wordDim; idx++) {
      dtype curValue = atof(vecInfo[idx + 1].c_str());
      sum[idx] = curValue;
      wordEmb[wordId][idx] = curValue;
    }

  } else {
    for (int idx = 0; idx < wordDim; idx++) {
      sum[idx] = 0.0;
    }
  }

  while (1) {
    if (!my_getline(inf, strLine)) {
      break;
    }
    if (strLine.empty())
      continue;
    split_bychar(strLine, vecInfo, ' ');
    if (vecInfo.size() != wordDim + 1) {
      std::cout << "error embedding file" << std::endl;
    }
    curWord = normalize_to_lowerwithdigit(vecInfo[0]);
    wordId = m_wordAlphabet.from_string(curWord);
    if (wordId >= 0) {
      count++;
      if (unknownId == wordId)
        bHasUnknown = true;
      indexers.insert(wordId);

      for (int idx = 0; idx < wordDim; idx++) {
        dtype curValue = atof(vecInfo[idx + 1].c_str());
        sum[idx] += curValue;
        wordEmb[wordId][idx] += curValue;
      }
    }

  }

  if (!bHasUnknown) {
    for (int idx = 0; idx < wordDim; idx++) {
      wordEmb[unknownId][idx] = sum[idx] / count;
    }
    count++;
    std::cout << unknownkey << " not found, using averaged value to initialize." << std::endl;
  }

  int oovWords = 0;
  int totalWords = 0;
  for (int id = 0; id < m_wordAlphabet.size(); id++) {
    if (indexers.find(id) == indexers.end()) {
      oovWords++;
      for (int idx = 0; idx < wordDim; idx++) {
        wordEmb[id][idx] = wordEmb[unknownId][idx];
      }
    }
    totalWords++;
  }

  std::cout << "OOV num is " << oovWords << ", total num is " << m_wordAlphabet.size() << ", embedding oov ratio is " << oovWords * 1.0 / m_wordAlphabet.size()
      << std::endl;

}

void Labeler::loadModelFile(const string& inputModelFile) {

}

void Labeler::writeModelFile(const string& outputModelFile) {

}

int main(int argc, char* argv[]) {
  std::string trainFile = "", devFile = "", testFile = "", modelFile = "";
  std::string wordEmbFile = "", charEmbFile = "", optionFile = "";
  std::string outputFile = "";
  bool bTrain = false;
  dsr::Argument_helper ah;

  ah.new_flag("l", "learn", "train or test", bTrain);
  ah.new_named_string("train", "trainCorpus", "named_string", "training corpus to train a model, must when training", trainFile);
  ah.new_named_string("dev", "devCorpus", "named_string", "development corpus to train a model, optional when training", devFile);
  ah.new_named_string("test", "testCorpus", "named_string",
      "testing corpus to train a model or input file to test a model, optional when training and must when testing", testFile);
  ah.new_named_string("model", "modelFile", "named_string", "model file, must when training and testing", modelFile);
  ah.new_named_string("word", "wordEmbFile", "named_string", "pretrained word embedding file to train a model, optional when training", wordEmbFile);
  ah.new_named_string("char", "charEmbFile", "named_string", "pretrained char embedding file to train a model, optional when training", charEmbFile);
  ah.new_named_string("option", "optionFile", "named_string", "option file to train a model, optional when training", optionFile);
  ah.new_named_string("output", "outputFile", "named_string", "output file to test, must when testing", outputFile);

  ah.process(argc, argv);

  Labeler tagger;
  if (bTrain) {
    tagger.train(trainFile, devFile, testFile, modelFile, optionFile, wordEmbFile, charEmbFile);
  } else {
    tagger.test(testFile, outputFile, modelFile);
  }

}
