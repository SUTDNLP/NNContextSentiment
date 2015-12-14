/*
 * GRCNNWordClassifier.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_GRCNNWordClassifier_H_
#define SRC_GRCNNWordClassifier_H_

#include <iostream>

#include <assert.h>
#include "Example.h"
#include "Feature.h"

#include "N3L.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

//A native neural network classfier using only word embeddings
template<typename xpu>
class GRCNNWordClassifier {
public:
	GRCNNWordClassifier() {
		_dropOut = 0.5;
	}
	~GRCNNWordClassifier() {

	}

public:
	LookupTable<xpu> _words;

	int _wordcontext;
	int _wordSize;
	int _wordDim;
	bool _b_wordEmb_finetune;
	int _wordHiddenSize;
	int _word_cnn_iSize;
	int _token_representation_size;

	int _hiddenSize;

	UniLayer<xpu> _tanh_project;
	UniLayer<xpu> _cnn_project;
	UniLayer<xpu> _olayer_linear;

	GRNN<xpu> _rnn_left;
	GRNN<xpu> _rnn_right;

	int _labelSize;

	Metric _eval;

	dtype _dropOut;

	int _remove; // 1, avg, 2, max, 3 min

	int _poolmanners;

public:

	inline void init(const NRMat<dtype>& wordEmb, int wordcontext, int labelSize, int wordHiddenSize, int hiddenSize) {
		_wordcontext = wordcontext;
		_wordSize = wordEmb.nrows();
		_wordDim = wordEmb.ncols();
		_poolmanners = 3;

		_labelSize = labelSize;
		_hiddenSize = hiddenSize;
		_wordHiddenSize = wordHiddenSize;
		_token_representation_size = _wordDim;

		_word_cnn_iSize = _token_representation_size * (2 * _wordcontext + 1);

		_words.initial(wordEmb);

		_rnn_left.initial(_wordHiddenSize, _word_cnn_iSize, true, 100);
		_rnn_right.initial(_wordHiddenSize, _word_cnn_iSize, false, 110);

		_cnn_project.initial(_wordHiddenSize, 2 * _wordHiddenSize, true, 20, 0);
		_tanh_project.initial(_hiddenSize, _poolmanners * _wordHiddenSize, true, 50, 0);
		_olayer_linear.initial(_labelSize, hiddenSize, false, 60, 2);

		_eval.reset();

		_remove = 0;

	}

	inline void release() {
		_words.release();

		_cnn_project.release();
		_tanh_project.release();
		_olayer_linear.release();
		_rnn_left.release();
		_rnn_right.release();
	}

	inline dtype process(const vector<Example>& examples, int iter) {
		_eval.reset();

		int example_num = examples.size();
		dtype cost = 0.0;
		int offset = 0;
		for (int count = 0; count < example_num; count++) {
			const Example& example = examples[count];

			int seq_size = example.m_features.size();
			if (seq_size > 2) {
				std::cout << "error" << std::endl;
			}

			Tensor<xpu, 3, dtype> wordprime, wordprimeLoss, wordprimeMask;
			Tensor<xpu, 3, dtype> wordrepresent, wordrepresentLoss;
			Tensor<xpu, 3, dtype> input, inputLoss;

			Tensor<xpu, 3, dtype> rnn_hidden_left_update;
			Tensor<xpu, 3, dtype> rnn_hidden_left_reset;
			Tensor<xpu, 3, dtype> rnn_hidden_left, rnn_hidden_leftLoss;
			Tensor<xpu, 3, dtype> rnn_hidden_left_afterreset;
			Tensor<xpu, 3, dtype> rnn_hidden_left_current;

			Tensor<xpu, 3, dtype> rnn_hidden_right_update;
			Tensor<xpu, 3, dtype> rnn_hidden_right_reset;
			Tensor<xpu, 3, dtype> rnn_hidden_right, rnn_hidden_rightLoss;
			Tensor<xpu, 3, dtype> rnn_hidden_right_afterreset;
			Tensor<xpu, 3, dtype> rnn_hidden_right_current;

			Tensor<xpu, 3, dtype> midhidden, midhiddenLoss;

			Tensor<xpu, 3, dtype> hidden, hiddenLoss;
			vector<Tensor<xpu, 2, dtype> > pool(_poolmanners), poolLoss(_poolmanners);
			vector<Tensor<xpu, 3, dtype> > poolIndex(_poolmanners);

			Tensor<xpu, 2, dtype> poolmerge, poolmergeLoss;
			Tensor<xpu, 2, dtype> project, projectLoss;
			Tensor<xpu, 2, dtype> output, outputLoss;

			//initialize
			int idx = seq_size - 1;

			{
				int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
				const Feature& feature = example.m_features[idx];
				int word_num = feature.words.size();
				int word_cnn_iSize = (idx == seq_size - 1) ? _word_cnn_iSize : _token_representation_size;
				int wordHiddenSize = (idx == seq_size - 1) ? _wordHiddenSize : _token_representation_size;

				wordprime = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), d_zero);
				wordprimeLoss = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), d_zero);
				wordprimeMask = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), d_one);
				wordrepresent = NewTensor<xpu>(Shape3(word_num, 1, _token_representation_size), d_zero);
				wordrepresentLoss = NewTensor<xpu>(Shape3(word_num, 1, _token_representation_size), d_zero);

				input = NewTensor<xpu>(Shape3(word_num, 1, word_cnn_iSize), d_zero);
				inputLoss = NewTensor<xpu>(Shape3(word_num, 1, word_cnn_iSize), d_zero);

				rnn_hidden_left_reset = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
				rnn_hidden_left_update = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
				rnn_hidden_left_afterreset = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
				rnn_hidden_left_current = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
				rnn_hidden_left = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
				rnn_hidden_leftLoss = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);

				rnn_hidden_right_reset = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
				rnn_hidden_right_update = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
				rnn_hidden_right_afterreset = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
				rnn_hidden_right_current = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
				rnn_hidden_right = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
				rnn_hidden_rightLoss = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);

				midhidden = NewTensor<xpu>(Shape3(word_num, 1, 2 * _wordHiddenSize), d_zero);
				midhiddenLoss = NewTensor<xpu>(Shape3(word_num, 1, 2 * _wordHiddenSize), d_zero);

				hidden = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), d_zero);
				hiddenLoss = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), d_zero);

				for (int idm = 0; idm < _poolmanners; idm++) {
					pool[idm] = NewTensor<xpu>(Shape2(1, wordHiddenSize), d_zero);
					poolLoss[idm] = NewTensor<xpu>(Shape2(1, wordHiddenSize), d_zero);
					poolIndex[idm] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), d_zero);
				}
			}

			poolmerge = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize), d_zero);
			poolmergeLoss = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize), d_zero);
			project = NewTensor<xpu>(Shape2(1, _hiddenSize), d_zero);
			projectLoss = NewTensor<xpu>(Shape2(1, _hiddenSize), d_zero);
			output = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);
			outputLoss = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);

			//forward propagation
			//input setting, and linear setting
			{
				const Feature& feature = example.m_features[idx];
				int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
				int curcontext = (idx == seq_size - 1) ? _wordcontext : 0;

				const vector<int>& words = feature.words;
				int word_num = words.size();
				//linear features should not be dropped out

				srand(iter * example_num + count * seq_size + idx);

				for (int idy = 0; idy < word_num; idy++) {
					_words.GetEmb(words[idy], wordprime[idy]);
				}

				//word dropout
				for (int idy = 0; idy < word_num; idy++) {
					dropoutcol(wordprimeMask[idy], _dropOut);
					wordprime[idy] = wordprime[idy] * wordprimeMask[idy];
				}

				//word representation
				for (int idy = 0; idy < word_num; idy++) {
					wordrepresent[idy] += wordprime[idy];
				}

				windowlized(wordrepresent, input, window);

				_rnn_left.ComputeForwardScore(input, rnn_hidden_left_reset, rnn_hidden_left_afterreset, rnn_hidden_left_update, rnn_hidden_left_current,
						rnn_hidden_left);
				_rnn_right.ComputeForwardScore(input, rnn_hidden_right_reset, rnn_hidden_right_afterreset, rnn_hidden_right_update, rnn_hidden_right_current,
						rnn_hidden_right);

				for (int idy = 0; idy < word_num; idy++) {
					concat(rnn_hidden_left[idy], rnn_hidden_right[idy], midhidden[idy]);
				}

				_cnn_project.ComputeForwardScore(midhidden, hidden);

				//word pooling
				if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
					avgpool_forward(hidden, pool[0], poolIndex[0]);
				}
				if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
					maxpool_forward(hidden, pool[1], poolIndex[1]);
				}
				if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
					minpool_forward(hidden, pool[2], poolIndex[2]);
				}
			}

			// sentence
			concat(pool, poolmerge);
			_tanh_project.ComputeForwardScore(poolmerge, project);
			_olayer_linear.ComputeForwardScore(project, output);

			cost += softmax_loss(output, example.m_labels, outputLoss, _eval, example_num);

			// loss backward propagation
			//sentence
			_olayer_linear.ComputeBackwardLoss(project, output, outputLoss, projectLoss);
			_tanh_project.ComputeBackwardLoss(poolmerge, project, projectLoss, poolmergeLoss);

			unconcat(poolLoss, poolmergeLoss);

			{

				const Feature& feature = example.m_features[idx];
				int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
				int curcontext = (idx == seq_size - 1) ? _wordcontext : 0;

				const vector<int>& words = feature.words;
				const vector<vector<int> >& chars = feature.chars;
				int word_num = words.size();

				//word pooling
				if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
					pool_backward(poolLoss[0], poolIndex[0], hiddenLoss);
				}
				if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
					pool_backward(poolLoss[1], poolIndex[1], hiddenLoss);
				}
				if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
					pool_backward(poolLoss[2], poolIndex[2], hiddenLoss);
				}

				_cnn_project.ComputeBackwardLoss(midhidden, hidden, hiddenLoss, midhiddenLoss);

				for (int idy = 0; idy < word_num; idy++) {
					unconcat(rnn_hidden_leftLoss[idy], rnn_hidden_rightLoss[idy], midhiddenLoss[idy]);
				}

				_rnn_left.ComputeBackwardLoss(input, rnn_hidden_left_reset, rnn_hidden_left_afterreset, rnn_hidden_left_update,
						rnn_hidden_left_current, rnn_hidden_left, rnn_hidden_leftLoss, inputLoss);
				_rnn_right.ComputeBackwardLoss(input, rnn_hidden_right_reset, rnn_hidden_right_afterreset, rnn_hidden_right_update,
						rnn_hidden_right_current, rnn_hidden_right, rnn_hidden_rightLoss, inputLoss);

				windowlized_backward(wordrepresentLoss, inputLoss, window);

				//word representation
				for (int idy = 0; idy < word_num; idy++) {
					wordprimeLoss[idy] += wordrepresentLoss[idy];
				}

				if (_words.bEmbFineTune()) {
					for (int idy = 0; idy < word_num; idy++) {
						wordprimeLoss[idy] = wordprimeLoss[idy] * wordprimeMask[idy];
						_words.EmbLoss(words[idy], wordprimeLoss[idy]);
					}
				}

			}

			//release
			{
				FreeSpace(&wordprime);
				FreeSpace(&wordprimeLoss);
				FreeSpace(&wordprimeMask);
				FreeSpace(&wordrepresent);
				FreeSpace(&wordrepresentLoss);
				FreeSpace(&input);
				FreeSpace(&inputLoss);
				FreeSpace(&hidden);
				FreeSpace(&hiddenLoss);
				for (int idm = 0; idm < _poolmanners; idm++) {
					FreeSpace(&(pool[idm]));
					FreeSpace(&(poolLoss[idm]));
					FreeSpace(&(poolIndex[idm]));
				}
			}

			FreeSpace(&rnn_hidden_left_reset);
			FreeSpace(&rnn_hidden_left_update);
			FreeSpace(&rnn_hidden_left_afterreset);
			FreeSpace(&rnn_hidden_left_current);
			FreeSpace(&rnn_hidden_left);
			FreeSpace(&rnn_hidden_leftLoss);

			FreeSpace(&rnn_hidden_right_reset);
			FreeSpace(&rnn_hidden_right_update);
			FreeSpace(&rnn_hidden_right_afterreset);
			FreeSpace(&rnn_hidden_right_current);
			FreeSpace(&rnn_hidden_right);
			FreeSpace(&rnn_hidden_rightLoss);

			FreeSpace(&midhidden);
			FreeSpace(&midhiddenLoss);

			FreeSpace(&poolmerge);
			FreeSpace(&poolmergeLoss);
			FreeSpace(&project);
			FreeSpace(&projectLoss);
			FreeSpace(&output);
			FreeSpace(&outputLoss);
		}

		if (_eval.getAccuracy() < 0) {
			std::cout << "strange" << std::endl;
		}

		return cost;
	}

	int predict(const vector<int>& linears, const vector<Feature>& features, vector<dtype>& results) {
		int seq_size = features.size();
		int offset = 0;
		if (seq_size > 2) {
			std::cout << "error" << std::endl;
		}

		Tensor<xpu, 3, dtype> wordprime;
		Tensor<xpu, 3, dtype> wordrepresent;
		Tensor<xpu, 3, dtype> input;

		Tensor<xpu, 3, dtype> rnn_hidden_left_update;
		Tensor<xpu, 3, dtype> rnn_hidden_left_reset;
		Tensor<xpu, 3, dtype> rnn_hidden_left;
		Tensor<xpu, 3, dtype> rnn_hidden_left_afterreset;
		Tensor<xpu, 3, dtype> rnn_hidden_left_current;

		Tensor<xpu, 3, dtype> rnn_hidden_right_update;
		Tensor<xpu, 3, dtype> rnn_hidden_right_reset;
		Tensor<xpu, 3, dtype> rnn_hidden_right;
		Tensor<xpu, 3, dtype> rnn_hidden_right_afterreset;
		Tensor<xpu, 3, dtype> rnn_hidden_right_current;

		Tensor<xpu, 3, dtype> midhidden;

		Tensor<xpu, 3, dtype> hidden;
		vector<Tensor<xpu, 2, dtype> > pool(_poolmanners);
		vector<Tensor<xpu, 3, dtype> > poolIndex(_poolmanners);

		Tensor<xpu, 2, dtype> poolmerge;
		Tensor<xpu, 2, dtype> project;
		Tensor<xpu, 2, dtype> output;

		//initialize
		int idx = seq_size - 1;

		{
			int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
			int word_cnn_iSize = (idx == seq_size - 1) ? _word_cnn_iSize : _token_representation_size;
			int wordHiddenSize = (idx == seq_size - 1) ? _wordHiddenSize : _token_representation_size;
			const Feature& feature = features[idx];
			int word_num = feature.words.size();

			wordprime = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), d_zero);
			wordrepresent = NewTensor<xpu>(Shape3(word_num, 1, _token_representation_size), d_zero);
			input = NewTensor<xpu>(Shape3(word_num, 1, word_cnn_iSize), d_zero);

			rnn_hidden_left_reset = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
			rnn_hidden_left_update = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
			rnn_hidden_left_afterreset = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
			rnn_hidden_left_current = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
			rnn_hidden_left = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);

			rnn_hidden_right_reset = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
			rnn_hidden_right_update = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
			rnn_hidden_right_afterreset = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
			rnn_hidden_right_current = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
			rnn_hidden_right = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);

			midhidden = NewTensor<xpu>(Shape3(word_num, 1, 2 * _wordHiddenSize), d_zero);

			hidden = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), d_zero);

			for (int idm = 0; idm < _poolmanners; idm++) {
				pool[idm] = NewTensor<xpu>(Shape2(1, wordHiddenSize), d_zero);
				poolIndex[idm] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), d_zero);
			}
		}
		poolmerge = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize), d_zero);
		project = NewTensor<xpu>(Shape2(1, _hiddenSize), d_zero);
		output = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);

		//forward propagation
		//input setting, and linear setting
		{
			const Feature& feature = features[idx];
			int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
			int curcontext = (idx == seq_size - 1) ? _wordcontext : 0;

			const vector<int>& words = feature.words;
			int word_num = words.size();
			//linear features should not be dropped out

			for (int idy = 0; idy < word_num; idy++) {
				_words.GetEmb(words[idy], wordprime[idy]);
			}

			//word representation
			for (int idy = 0; idy < word_num; idy++) {
				wordrepresent[idy] += wordprime[idy];
			}

			windowlized(wordrepresent, input, window);

			_rnn_left.ComputeForwardScore(input, rnn_hidden_left_reset, rnn_hidden_left_afterreset, rnn_hidden_left_update, rnn_hidden_left_current,
					rnn_hidden_left);
			_rnn_right.ComputeForwardScore(input, rnn_hidden_right_reset, rnn_hidden_right_afterreset, rnn_hidden_right_update, rnn_hidden_right_current,
					rnn_hidden_right);

			for (int idy = 0; idy < word_num; idy++) {
				concat(rnn_hidden_left[idy], rnn_hidden_right[idy], midhidden[idy]);
			}

			_cnn_project.ComputeForwardScore(midhidden, hidden);

			//word pooling
			if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
				avgpool_forward(hidden, pool[0], poolIndex[0]);
			}
			if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
				maxpool_forward(hidden, pool[1], poolIndex[1]);
			}
			if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
				minpool_forward(hidden, pool[2], poolIndex[2]);
			}
		}

		// sentence
		concat(pool, poolmerge);
		_tanh_project.ComputeForwardScore(poolmerge, project);
		_olayer_linear.ComputeForwardScore(project, output);

		// decode algorithm
		int optLabel = softmax_predict(output, results);

		//release
		{
			FreeSpace(&wordprime);
			FreeSpace(&wordrepresent);
			FreeSpace(&input);

			FreeSpace(&rnn_hidden_left_reset);
			FreeSpace(&rnn_hidden_left_update);
			FreeSpace(&rnn_hidden_left_afterreset);
			FreeSpace(&rnn_hidden_left_current);
			FreeSpace(&rnn_hidden_left);

			FreeSpace(&rnn_hidden_right_reset);
			FreeSpace(&rnn_hidden_right_update);
			FreeSpace(&rnn_hidden_right_afterreset);
			FreeSpace(&rnn_hidden_right_current);
			FreeSpace(&rnn_hidden_right);

			FreeSpace(&midhidden);

			FreeSpace(&hidden);
			for (int idm = 0; idm < _poolmanners; idm++) {
				FreeSpace(&(pool[idm]));
				FreeSpace(&(poolIndex[idm]));
			}
		}
		FreeSpace(&poolmerge);
		FreeSpace(&project);
		FreeSpace(&output);

		return optLabel;
	}

	dtype computeScore(const Example& example) {
		int seq_size = example.m_features.size();
		int offset = 0;
		if (seq_size > 2) {
			std::cout << "error" << std::endl;
		}

		Tensor<xpu, 3, dtype> wordprime;
		Tensor<xpu, 3, dtype> wordrepresent;
		Tensor<xpu, 3, dtype> input;

		Tensor<xpu, 3, dtype> rnn_hidden_left_update;
		Tensor<xpu, 3, dtype> rnn_hidden_left_reset;
		Tensor<xpu, 3, dtype> rnn_hidden_left;
		Tensor<xpu, 3, dtype> rnn_hidden_left_afterreset;
		Tensor<xpu, 3, dtype> rnn_hidden_left_current;

		Tensor<xpu, 3, dtype> rnn_hidden_right_update;
		Tensor<xpu, 3, dtype> rnn_hidden_right_reset;
		Tensor<xpu, 3, dtype> rnn_hidden_right;
		Tensor<xpu, 3, dtype> rnn_hidden_right_afterreset;
		Tensor<xpu, 3, dtype> rnn_hidden_right_current;

		Tensor<xpu, 3, dtype> midhidden;

		Tensor<xpu, 3, dtype> hidden;
		vector<Tensor<xpu, 2, dtype> > pool(_poolmanners);
		vector<Tensor<xpu, 3, dtype> > poolIndex(_poolmanners);

		Tensor<xpu, 2, dtype> poolmerge;
		Tensor<xpu, 2, dtype> project;
		Tensor<xpu, 2, dtype> output;

		//initialize
		int idx = seq_size - 1;

		{
			int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
			int word_cnn_iSize = (idx == seq_size - 1) ? _word_cnn_iSize : _token_representation_size;
			int wordHiddenSize = (idx == seq_size - 1) ? _wordHiddenSize : _token_representation_size;
			const Feature& feature = example.m_features[idx];
			int word_num = feature.words.size();

			wordprime = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), d_zero);
			wordrepresent = NewTensor<xpu>(Shape3(word_num, 1, _token_representation_size), d_zero);
			input = NewTensor<xpu>(Shape3(word_num, 1, word_cnn_iSize), d_zero);

			rnn_hidden_left_reset = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
			rnn_hidden_left_update = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
			rnn_hidden_left_afterreset = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
			rnn_hidden_left_current = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
			rnn_hidden_left = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);

			rnn_hidden_right_reset = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
			rnn_hidden_right_update = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
			rnn_hidden_right_afterreset = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
			rnn_hidden_right_current = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
			rnn_hidden_right = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);

			midhidden = NewTensor<xpu>(Shape3(word_num, 1, 2 * _wordHiddenSize), d_zero);

			hidden = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), d_zero);

			for (int idm = 0; idm < _poolmanners; idm++) {
				pool[idm] = NewTensor<xpu>(Shape2(1, wordHiddenSize), d_zero);
				poolIndex[idm] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), d_zero);
			}
		}

		poolmerge = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize), d_zero);
		project = NewTensor<xpu>(Shape2(1, _hiddenSize), d_zero);
		output = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);

		//forward propagation
		//input setting, and linear setting
		{
			const Feature& feature = example.m_features[idx];
			int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
			int curcontext = (idx == seq_size - 1) ? _wordcontext : 0;

			const vector<int>& words = feature.words;
			int word_num = words.size();
			//linear features should not be dropped out

			for (int idy = 0; idy < word_num; idy++) {
				_words.GetEmb(words[idy], wordprime[idy]);
			}

			//word representation
			for (int idy = 0; idy < word_num; idy++) {
				wordrepresent[idy] += wordprime[idy];
			}

			windowlized(wordrepresent, input, window);

			_rnn_left.ComputeForwardScore(input, rnn_hidden_left_reset, rnn_hidden_left_afterreset, rnn_hidden_left_update, rnn_hidden_left_current,
					rnn_hidden_left);
			_rnn_right.ComputeForwardScore(input, rnn_hidden_right_reset, rnn_hidden_right_afterreset, rnn_hidden_right_update, rnn_hidden_right_current,
					rnn_hidden_right);

			for (int idy = 0; idy < word_num; idy++) {
				concat(rnn_hidden_left[idy], rnn_hidden_right[idy], midhidden[idy]);
			}

			_cnn_project.ComputeForwardScore(midhidden, hidden);

			//word pooling
			if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
				avgpool_forward(hidden, pool[0], poolIndex[0]);
			}
			if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
				maxpool_forward(hidden, pool[1], poolIndex[1]);
			}
			if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
				minpool_forward(hidden, pool[2], poolIndex[2]);
			}
		}

		// sentence
		concat(pool, poolmerge);
		_tanh_project.ComputeForwardScore(poolmerge, project);
		_olayer_linear.ComputeForwardScore(project, output);

		dtype cost = softmax_cost(output, example.m_labels);

		//release
		{
			FreeSpace(&wordprime);
			FreeSpace(&wordrepresent);
			FreeSpace(&input);

			FreeSpace(&rnn_hidden_left_reset);
			FreeSpace(&rnn_hidden_left_update);
			FreeSpace(&rnn_hidden_left_afterreset);
			FreeSpace(&rnn_hidden_left_current);
			FreeSpace(&rnn_hidden_left);

			FreeSpace(&rnn_hidden_right_reset);
			FreeSpace(&rnn_hidden_right_update);
			FreeSpace(&rnn_hidden_right_afterreset);
			FreeSpace(&rnn_hidden_right_current);
			FreeSpace(&rnn_hidden_right);

			FreeSpace(&midhidden);

			FreeSpace(&hidden);
			for (int idm = 0; idm < _poolmanners; idm++) {
				FreeSpace(&(pool[idm]));
				FreeSpace(&(poolIndex[idm]));
			}
		}
		FreeSpace(&poolmerge);
		FreeSpace(&project);
		FreeSpace(&output);

		return cost;
	}

	void updateParams(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
		_cnn_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_tanh_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_olayer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_rnn_left.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_rnn_right.updateAdaGrad(nnRegular, adaAlpha, adaEps);

		_words.updateAdaGrad(nnRegular, adaAlpha, adaEps);

	}

	void writeModel();

	void loadModel();

	void checkgrads(const vector<Example>& examples, int iter) {

		checkgrad(this, examples, _olayer_linear._W, _olayer_linear._gradW, "_olayer_linear._W", iter);
		checkgrad(this, examples, _olayer_linear._b, _olayer_linear._gradb, "_olayer_linear._b", iter);

		checkgrad(this, examples, _tanh_project._W, _tanh_project._gradW, "_tanh_project._W", iter);
		checkgrad(this, examples, _tanh_project._b, _tanh_project._gradb, "_tanh_project._b", iter);

		checkgrad(this, examples, _cnn_project._W, _cnn_project._gradW, "_cnn_project._W", iter);
		checkgrad(this, examples, _cnn_project._b, _cnn_project._gradb, "_cnn_project._b", iter);

		checkgrad(this, examples, _rnn_left._rnn_update._WL, _rnn_left._rnn_update._gradWL, "_rnn_left._rnn_update._WL", iter);
		checkgrad(this, examples, _rnn_left._rnn_update._WR, _rnn_left._rnn_update._gradWR, "_rnn_left._rnn_update._WR", iter);
		checkgrad(this, examples, _rnn_left._rnn_update._b, _rnn_left._rnn_update._gradb, "_rnn_left._rnn_update._b", iter);

		checkgrad(this, examples, _rnn_right._rnn_update._WL, _rnn_right._rnn_update._gradWL, "_rnn_right._rnn_update._WL", iter);
		checkgrad(this, examples, _rnn_right._rnn_update._WR, _rnn_right._rnn_update._gradWR, "_rnn_right._rnn_update._WR", iter);
		checkgrad(this, examples, _rnn_right._rnn_update._b, _rnn_right._rnn_update._gradb, "_rnn_right._rnn_update._b", iter);

		checkgrad(this, examples, _rnn_left._rnn_reset._WL, _rnn_left._rnn_reset._gradWL, "_rnn_left._rnn_reset._WL", iter);
		checkgrad(this, examples, _rnn_left._rnn_reset._WR, _rnn_left._rnn_reset._gradWR, "_rnn_left._rnn_reset._WR", iter);
		checkgrad(this, examples, _rnn_left._rnn_reset._b, _rnn_left._rnn_reset._gradb, "_rnn_left._rnn_reset._b", iter);

		checkgrad(this, examples, _rnn_right._rnn_reset._WL, _rnn_right._rnn_reset._gradWL, "_rnn_right._rnn_reset._WL", iter);
		checkgrad(this, examples, _rnn_right._rnn_reset._WR, _rnn_right._rnn_reset._gradWR, "_rnn_right._rnn_reset._WR", iter);
		checkgrad(this, examples, _rnn_right._rnn_reset._b, _rnn_right._rnn_reset._gradb, "_rnn_right._rnn_reset._b", iter);

		checkgrad(this, examples, _rnn_left._rnn._WL, _rnn_left._rnn._gradWL, "_rnn_left._rnn._WL", iter);
		checkgrad(this, examples, _rnn_left._rnn._WR, _rnn_left._rnn._gradWR, "_rnn_left._rnn._WR", iter);
		checkgrad(this, examples, _rnn_left._rnn._b, _rnn_left._rnn._gradb, "_rnn_left._rnn._b", iter);

		checkgrad(this, examples, _rnn_right._rnn._WL, _rnn_right._rnn._gradWL, "_rnn_right._rnn._WL", iter);
		checkgrad(this, examples, _rnn_right._rnn._WR, _rnn_right._rnn._gradWR, "_rnn_right._rnn._WR", iter);
		checkgrad(this, examples, _rnn_right._rnn._b, _rnn_right._rnn._gradb, "_rnn_right._rnn._b", iter);

		checkgrad(this, examples, _words._E, _words._gradE, "_words._E", iter, _words._indexers);

	}

public:
	inline void resetEval() {
		_eval.reset();
	}

	inline void setDropValue(dtype dropOut) {
		_dropOut = dropOut;
	}

	inline void setWordEmbFinetune(bool b_wordEmb_finetune) {
		_b_wordEmb_finetune = b_wordEmb_finetune;
	}

	inline void resetRemove(int remove) {
		_remove = remove;
	}

};

#endif /* SRC_GRCNNWordClassifier_H_ */
