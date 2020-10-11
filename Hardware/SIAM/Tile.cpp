/*******************************************************************************
* Copyright (c) 2017-2020
* School of Electrical, Computer and Energy Engineering, Arizona State University
* PI: Prof. Yu Cao
* All rights reserved.
*
* This source code is part of NeuroSim - a device-circuit-algorithm framework to benchmark
* neuro-inspired architectures with synaptic devices(e.g., SRAM and emerging non-volatile memory).
* Copyright of the model is maintained by the developers, and the model is distributed under
* the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License
* http://creativecommons.org/licenses/by-nc/4.0/legalcode.
* The source code is free and you can redistribute and/or modify it
* by providing that the following conditions are met:
*
*  1) Redistributions of source code must retain the above copyright notice,
*     this list of conditions and the following disclaimer.
*
*  2) Redistributions in binary form must reproduce the above copyright notice,
*     this list of conditions and the following disclaimer in the documentation
*     and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* Developer list:
*		Gokul Krishnan Email: gkrish19@asu.edu

* Credits: Prof.Shimeng Yu and his research group for NeuroSim
********************************************************************************/

#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <sstream>
#include "Sigmoid.h"
#include "BitShifter.h"
#include "AdderTree.h"
#include "Buffer.h"
#include "HTree.h"
#include "ProcessingUnit.h"
#include "SubArray.h"
#include "constant.h"
#include "formula.h"
#include "Param.h"
#include "Tile.h"

using namespace std;

extern Param *param;

SubArray *subArrayInPE;
Buffer *inputBuffer;
Buffer *outputBuffer;
HTree *hTree;
AdderTree *accumulation;
Sigmoid *sigmoid;
BitShifter *reLu;


void TileInitialize_new(InputParameter& inputParameter, Technology& tech, MemCell& cell, double _numPE, double _peSize_x, double _peSize_y){

	subArrayInPE = new SubArray(inputParameter, tech, cell);
	inputBuffer = new Buffer(inputParameter, tech, cell);
	outputBuffer = new Buffer(inputParameter, tech, cell);
	hTree = new HTree(inputParameter, tech, cell);
	accumulation = new AdderTree(inputParameter, tech, cell);

	//cout<<"\n _numPE is"<<_numPE<<endl;

	if (!param->chipActivation) {
		if (param->reLu) {
			reLu = new BitShifter(inputParameter, tech, cell);
		} else {
			sigmoid = new Sigmoid(inputParameter, tech, cell);
		}
	}

	/*** Parameters ***/
	double numPE, peSize_x, peSize_y, numSubArray;
	int numRowPerSynapse, numColPerSynapse;

	numPE = _numPE;
	peSize_x = _peSize_x;
	peSize_y = _peSize_y;
	numRowPerSynapse = param->numRowPerSynapse;
	numColPerSynapse = param->numColPerSynapse;

	/*** Initialize ProcessingUnit ***/
	numSubArray = ceil((double)peSize_x/(double)param->numRowSubArray)*ceil((double)peSize_y/(double)param->numColSubArray);
	//cout<<"numSubArray is:"<<numSubArray<<endl;
	ProcessingUnitInitialize(subArrayInPE, inputParameter, tech, cell, ceil(sqrt(numSubArray)), ceil(sqrt(numSubArray)));

	if (param->parallelRead) {
		accumulation->Initialize((int) ceil((double)sqrt(numPE)), ceil((double)log2((double)param->levelOutput))+param->numBitInput+1+ceil((double)log2((double)peSize_x/(double)param->numRowSubArray)),
								ceil((double)sqrt((double)numPE)*(double)param->numColSubArray/(double)param->numColMuxed));
		if (!param->chipActivation) {
			if (param->reLu) {
				reLu->Initialize(ceil((double)peSize_y*(double)param->numColSubArray/(double)param->numColMuxed), param->numBitInput, param->clkFreq);
			} else {
				sigmoid->Initialize(false, param->numBitInput, ceil((double)log2((double)param->levelOutput))+param->numBitInput+1+ceil((double)log2((double)peSize_x/(double)param->numRowSubArray))+ceil((double)log2((double)sqrt((double)numPE))),
								ceil((double)sqrt((double)numPE)*(double)param->numColSubArray/(double)param->numColMuxed), param->clkFreq);
			}
			outputBuffer->Initialize(param->numBitInput*numPE*param->numColSubArray/param->numColMuxed, param->numBitInput*numPE, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
		} else {
			outputBuffer->Initialize((ceil((double)log2((double)param->levelOutput))+param->numBitInput+1+ceil((double)log2((double)peSize_x/(double)param->numRowSubArray)))*numPE*param->numColSubArray/param->numColMuxed,
								(ceil((double)log2((double)param->levelOutput))+param->numBitInput+1+ceil((double)log2((double)peSize_x/(double)param->numRowSubArray)))*numPE,
								1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
		}
	} else {
		accumulation->Initialize((int) ceil((double)sqrt((double)numPE)), ceil((double)log2((double)param->numRowSubArray)+(double)param->cellBit-1)+param->numBitInput+1+ceil((double)log2((double)peSize_x/(double)param->numRowSubArray)),
								ceil((double)sqrt(numPE)*(double)param->numColSubArray/(double)param->numColMuxed));
		if (!param->chipActivation) {
			if (param->reLu) {
				reLu->Initialize(ceil((double)peSize_y*(double)param->numColSubArray/(double)param->numColMuxed), param->numBitInput, param->clkFreq);
			} else {
				sigmoid->Initialize(false, param->numBitInput, ceil((double)log2((double)param->numRowSubArray)+(double)param->cellBit-1)+param->numBitInput+1+ceil((double)log2((double)peSize_x/(double)param->numRowSubArray))+ceil((double)log2((double)sqrt((double)numPE))),
								ceil((double)sqrt((double)numPE)*(double)param->numColSubArray/(double)param->numColMuxed), param->clkFreq);
			}
			outputBuffer->Initialize(param->numBitInput*numPE*param->numColSubArray/param->numColMuxed, param->numBitInput*numPE, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
		} else {
			outputBuffer->Initialize((ceil((double)log2((double)param->numRowSubArray)+(double)param->cellBit-1)+param->numBitInput+1+ceil((double)log2((double)peSize_x/(double)param->numRowSubArray)))*numPE*param->numColSubArray/param->numColMuxed,
								(ceil((double)log2((double)param->numRowSubArray)+(double)param->cellBit-1)+param->numBitInput+1+ceil((double)log2((double)peSize_x/(double)param->numRowSubArray)))*numPE,
								1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
		}
	}

	inputBuffer->Initialize(param->numBitInput*param->numRowSubArray, numPE, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
	hTree->Initialize(ceil((double)sqrt((double)numPE)), ceil((double)sqrt((double)numPE)), param->localBusDelayTolerance, numPE*param->numRowSubArray);
}


void TileInitialize(InputParameter& inputParameter, Technology& tech, MemCell& cell, double _numPE, double _peSize){

	subArrayInPE = new SubArray(inputParameter, tech, cell);
	inputBuffer = new Buffer(inputParameter, tech, cell);
	outputBuffer = new Buffer(inputParameter, tech, cell);
	hTree = new HTree(inputParameter, tech, cell);
	accumulation = new AdderTree(inputParameter, tech, cell);

	if (!param->chipActivation) {
		if (param->reLu) {
			reLu = new BitShifter(inputParameter, tech, cell);
		} else {
			sigmoid = new Sigmoid(inputParameter, tech, cell);
		}
	}

	/*** Parameters ***/
	double numPE, peSize, numSubArray;
	int numRowPerSynapse, numColPerSynapse;

	numPE = _numPE -2;
	peSize = _peSize;
	numRowPerSynapse = param->numRowPerSynapse;
	numColPerSynapse = param->numColPerSynapse;

	/*** Initialize ProcessingUnit ***/
	numSubArray = ceil((double)peSize/(double)param->numRowSubArray)*ceil((double)peSize/(double)param->numColSubArray);
	ProcessingUnitInitialize(subArrayInPE, inputParameter, tech, cell, ceil(sqrt(numSubArray)), ceil(sqrt(numSubArray)));

	if (param->parallelRead) {
		accumulation->Initialize((int) ceil((double)sqrt(numPE)), ceil((double)log2((double)param->levelOutput))+param->numBitInput+1+ceil((double)log2((double)peSize/(double)param->numRowSubArray)),
								ceil((double)sqrt((double)numPE)*(double)param->numColSubArray/(double)param->numColMuxed));
		if (!param->chipActivation) {
			if (param->reLu) {
				reLu->Initialize(ceil((double)peSize*(double)param->numColSubArray/(double)param->numColMuxed), param->numBitInput, param->clkFreq);
			} else {
				sigmoid->Initialize(false, param->numBitInput, ceil((double)log2((double)param->levelOutput))+param->numBitInput+1+ceil((double)log2((double)peSize/(double)param->numRowSubArray))+ceil((double)log2((double)sqrt((double)numPE))),
								ceil((double)sqrt((double)numPE)*(double)param->numColSubArray/(double)param->numColMuxed), param->clkFreq);
			}
			outputBuffer->Initialize(param->numBitInput*numPE*param->numColSubArray/param->numColMuxed, param->numBitInput*numPE, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
		} else {
			outputBuffer->Initialize((ceil((double)log2((double)param->levelOutput))+param->numBitInput+1+ceil((double)log2((double)peSize/(double)param->numRowSubArray)))*numPE*param->numColSubArray/param->numColMuxed,
								(ceil((double)log2((double)param->levelOutput))+param->numBitInput+1+ceil((double)log2((double)peSize/(double)param->numRowSubArray)))*numPE,
								1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
		}
	} else {
		accumulation->Initialize((int) ceil((double)sqrt((double)numPE)), ceil((double)log2((double)param->numRowSubArray)+(double)param->cellBit-1)+param->numBitInput+1+ceil((double)log2((double)peSize/(double)param->numRowSubArray)),
								ceil((double)sqrt(numPE)*(double)param->numColSubArray/(double)param->numColMuxed));
		if (!param->chipActivation) {
			if (param->reLu) {
				reLu->Initialize(ceil((double)peSize*(double)param->numColSubArray/(double)param->numColMuxed), param->numBitInput, param->clkFreq);
			} else {
				sigmoid->Initialize(false, param->numBitInput, ceil((double)log2((double)param->numRowSubArray)+(double)param->cellBit-1)+param->numBitInput+1+ceil((double)log2((double)peSize/(double)param->numRowSubArray))+ceil((double)log2((double)sqrt((double)numPE))),
								ceil((double)sqrt((double)numPE)*(double)param->numColSubArray/(double)param->numColMuxed), param->clkFreq);
			}
			outputBuffer->Initialize(param->numBitInput*numPE*param->numColSubArray/param->numColMuxed, param->numBitInput*numPE, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
		} else {
			outputBuffer->Initialize((ceil((double)log2((double)param->numRowSubArray)+(double)param->cellBit-1)+param->numBitInput+1+ceil((double)log2((double)peSize/(double)param->numRowSubArray)))*numPE*param->numColSubArray/param->numColMuxed,
								(ceil((double)log2((double)param->numRowSubArray)+(double)param->cellBit-1)+param->numBitInput+1+ceil((double)log2((double)peSize/(double)param->numRowSubArray)))*numPE,
								1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
		}
	}

	inputBuffer->Initialize(param->numBitInput*param->numRowSubArray, numPE, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
	hTree->Initialize(ceil((double)sqrt((double)numPE)), ceil((double)sqrt((double)numPE)), param->localBusDelayTolerance, numPE*param->numRowSubArray);
}


vector<double> TileCalculateArea(double numPE, double peSize, double *height, double *width) {
	double area = 0;
	double PEheight, PEwidth, PEbufferArea;
	*height = 0;
	*width = 0;
	vector<double> areaResults;
	vector<double> peAreaResults;

	int numSubArray = ceil((double) peSize/(double) param->numRowSubArray)*ceil((double) peSize/(double) param->numColSubArray);
	peAreaResults = ProcessingUnitCalculateArea(subArrayInPE, ceil((double)sqrt((double)numSubArray)), ceil((double)sqrt((double)numSubArray)), &PEheight, &PEwidth, &PEbufferArea);
	double PEarea = peAreaResults[0];
	double PEareaADC = peAreaResults[1];
	double PEareaAccum = peAreaResults[2];
	double PEareaOther = peAreaResults[3];
	double PEareabuffer = peAreaResults[4];
	double PEareaotheralone = peAreaResults[5];
	double PEsubarrayarea_alone = peAreaResults[6];

	accumulation->CalculateArea(NULL, ceil((double)sqrt((double)numPE))*PEwidth, NONE);
	if (!param->chipActivation) {
		if (param->reLu) {
			reLu->CalculateArea(NULL, ceil((double)sqrt((double)numPE))*PEwidth, NONE);
			area += reLu->area;
		} else {
			sigmoid->CalculateUnitArea(NONE);
			sigmoid->CalculateArea(NULL, ceil((double)sqrt((double)numPE))*PEwidth, NONE);
			area += sigmoid->area;
		}
	}
	inputBuffer->CalculateArea(ceil((double)sqrt((double)numPE))*PEheight, NULL, NONE);
	outputBuffer->CalculateArea(NULL, ceil((double)sqrt((double)numPE))*PEwidth, NONE);
	hTree->CalculateArea(PEheight, PEwidth, 16);

	area += PEarea*numPE + accumulation->area + inputBuffer->area + outputBuffer->area + hTree->area;

	*height = sqrt(area);
	*width = area/(*height);

	areaResults.push_back(area);
	areaResults.push_back(hTree->area);
	areaResults.push_back(PEareaADC*numPE);
	areaResults.push_back(PEareaAccum*numPE + accumulation->area);
	areaResults.push_back(PEareaOther*numPE);
	areaResults.push_back(PEareabuffer*numPE + inputBuffer->area + outputBuffer->area);
	areaResults.push_back(PEareaOther*numPE);
	areaResults.push_back(PEsubarrayarea_alone*numPE);

	return areaResults;
}

vector<double> TileCalculateArea_new(double numPE, double peSize_x, double peSize_y, double *height, double *width) {
	double area = 0;
	double PEheight, PEwidth, PEbufferArea;
	*height = 0;
	*width = 0;
	vector<double> areaResults;
	vector<double> peAreaResults;
	//cout<<"\n numPE is "<<numPE<<endl;
	int numSubArray = ceil((double) peSize_x/(double) param->numRowSubArray)*ceil((double) peSize_y/(double) param->numColSubArray);
	peAreaResults = ProcessingUnitCalculateArea(subArrayInPE, ceil((double) peSize_x/(double) param->numRowSubArray), ceil((double) peSize_y/(double) param->numColSubArray), &PEheight, &PEwidth, &PEbufferArea);
	double PEarea = peAreaResults[0];
	double PEareaADC = peAreaResults[1];
	double PEareaAccum = peAreaResults[2];
	double PEareaOther = peAreaResults[3];
	double PEareabuffer = peAreaResults[4];
	double PEareaotheralone = peAreaResults[5];
	double PEsubarrayarea_alone = peAreaResults[6];
/*
	cout<<"Total PE area is is "<<PEarea*1e12<< " um^2"<<endl;
	cout<<"Total PE ADC area is is "<<PEareaADC*1e12<< " um^2"<<endl;
	cout<<"Total PE Accum area is is "<<PEareaAccum*1e12<< " um^2"<<endl;
	cout<<"Total PE Other area is is "<<PEareaOther*1e12<< " um^2"<<endl;
	cout<<"Total PE Subarray area is is "<<PEsubarrayarea_alone*1e12<< " um^2"<<endl;

*/
	accumulation->CalculateArea(NULL, ceil((double)sqrt((double)numPE))*PEwidth, NONE);
	if (!param->chipActivation) {
		if (param->reLu) {
			reLu->CalculateArea(NULL, ceil((double)sqrt((double)numPE))*PEwidth, NONE);
			area += reLu->area;
		} else {
			sigmoid->CalculateUnitArea(NONE);
			sigmoid->CalculateArea(NULL, ceil((double)sqrt((double)numPE))*PEwidth, NONE);
			area += sigmoid->area;
		}
	}
	inputBuffer->CalculateArea(ceil((double)sqrt((double)numPE))*PEheight, NULL, NONE);
	outputBuffer->CalculateArea(NULL, ceil((double)sqrt((double)numPE))*PEwidth, NONE);
	hTree->CalculateArea(PEheight, PEwidth, 16);

	area = PEarea*numPE + accumulation->area + inputBuffer->area + outputBuffer->area + hTree->area;

	*height = sqrt(area);
	*width = area/(*height);

	areaResults.push_back(area);
	areaResults.push_back(hTree->area);
	areaResults.push_back(PEareaADC*numPE);
	areaResults.push_back(PEareaAccum*numPE + accumulation->area);
	areaResults.push_back(PEareaOther*numPE);
	areaResults.push_back(PEareabuffer*numPE + inputBuffer->area + outputBuffer->area);
	//areaResults.push_back(PEareaOther*numPE);
	areaResults.push_back(PEsubarrayarea_alone*numPE);

	return areaResults;
}


void TileCalculatePerformance(const vector<vector<double> > &newMemory, const vector<vector<double> > &oldMemory, const vector<vector<double> > &inputVector, int novelMap, double numPE,
							double peSize, int speedUpRow, int speedUpCol, int weightMatrixRow, int weightMatrixCol, int numInVector, MemCell& cell, double *readLatency, double *readDynamicEnergy, double *leakage,
							double *bufferLatency, double *bufferDynamicEnergy, double *icLatency, double *icDynamicEnergy,
							double *coreLatencyADC, double *coreLatencyAccum, double *coreLatencyOther, double *coreEnergyADC, double *coreEnergyAccum, double *coreEnergyOther) {

	/*** sweep PE ***/
	int numRowPerSynapse, numColPerSynapse;
	numRowPerSynapse = param->numRowPerSynapse;
	numColPerSynapse = param->numColPerSynapse;
	double PEreadLatency, PEreadDynamicEnergy, PEleakage, PEbufferLatency, PEbufferDynamicEnergy, PEicLatency, PEicDynamicEnergy;
	double peLatencyADC, peLatencyAccum, peLatencyOther, peEnergyADC, peEnergyAccum, peEnergyOther;
	int numSubArrayRow = ceil((double)peSize/(double)param->numRowSubArray);
	int numSubArrayCol = ceil((double)peSize/(double)param->numColSubArray);

	*readLatency = 0;
	*readDynamicEnergy = 0;
	*leakage = 0;
	*bufferLatency = 0;
	*bufferDynamicEnergy = 0;
	*icLatency = 0;
	*icDynamicEnergy = 0;
	*coreEnergyADC = 0;
	*coreEnergyAccum = 0;
	*coreEnergyOther = 0;
	*coreLatencyADC = 0;
	*coreLatencyAccum = 0;
	*coreLatencyOther = 0;

	if (!novelMap) {   // conventional Mapping
		if (speedUpRow*speedUpCol > 1) {
			if ( (speedUpRow >= ceil(sqrt((double)numPE))) && (speedUpCol >= ceil(sqrt((double)numPE))) ) {
				// duplication in PE or subArray --> tell each PE to take the whole assigned weight  --> "fully" duplication
				// assign weight and input to specific tile
				vector<vector<double> > pEMemory;
				pEMemory = CopyPEArray(newMemory, 0, 0, weightMatrixRow, weightMatrixCol);
				vector<vector<double> > pEInput;
				pEInput = CopyPEInput(inputVector, 0, numInVector, weightMatrixRow);

				ProcessingUnitCalculatePerformance(subArrayInPE, pEMemory, pEMemory, pEInput, ceil((double)speedUpRow/sqrt((double)numPE)), ceil((double)speedUpCol/sqrt((double)numPE)),
											numSubArrayRow, numSubArrayCol, weightMatrixRow, weightMatrixCol, numInVector, cell, &PEreadLatency, &PEreadDynamicEnergy, &PEleakage,
											&PEbufferLatency, &PEbufferDynamicEnergy, &PEicLatency, &PEicDynamicEnergy,
											&peLatencyADC, &peLatencyAccum, &peLatencyOther, &peEnergyADC, &peEnergyAccum, &peEnergyOther);

				*readLatency = PEreadLatency/numPE;  // further speed up in PE level
				*readDynamicEnergy = PEreadDynamicEnergy;   // since subArray.cpp takes all input vectors, no need to *numPE here
				*bufferLatency = PEbufferLatency/numPE;
				*bufferDynamicEnergy = PEbufferDynamicEnergy;
				*icLatency = PEicLatency/numPE;
				*icDynamicEnergy = PEicDynamicEnergy;

				*coreLatencyADC = peLatencyADC/numPE;
				*coreLatencyAccum = peLatencyAccum/numPE;
				*coreLatencyOther = peLatencyOther/numPE;

				*coreEnergyADC = peEnergyADC;
				*coreEnergyAccum = peEnergyAccum;
				*coreEnergyOther = peEnergyOther;
				// no accumulation access
			} else {
				// # duplication is smaller then # PE, means only a group of PE take the assigned weight  --> not "fully" duplication
				// also need to redefine a few data-grab start-point
				for (int i=0; i<ceil((double)weightMatrixRow/(double)peSize); i++) {
					for (int j=0; j<ceil((double)weightMatrixCol/(double)peSize); j++) {
						if ( (i*peSize < weightMatrixRow) && (j*peSize < weightMatrixCol) ) {
							int numRowMatrix = min(peSize, (double) weightMatrixRow-i*peSize);
							int numColMatrix = min(peSize, (double) weightMatrixCol-j*peSize);

							// assign weight and input to specific tile
							vector<vector<double> > pEMemory;
							pEMemory = CopyPEArray(newMemory, i*peSize, j*peSize, numRowMatrix, numColMatrix);
							vector<vector<double> > pEInput;
							pEInput = CopyPEInput(inputVector, i*peSize, numInVector, numRowMatrix);

							ProcessingUnitCalculatePerformance(subArrayInPE, pEMemory, pEMemory, pEInput, ceil((double)speedUpRow/sqrt((double)numPE)), ceil((double)speedUpCol/sqrt((double)numPE)),
												numSubArrayRow, numSubArrayCol, numRowMatrix, numColMatrix, numInVector, cell, &PEreadLatency, &PEreadDynamicEnergy, &PEleakage,
												&PEbufferLatency, &PEbufferDynamicEnergy, &PEicLatency, &PEicDynamicEnergy,
												&peLatencyADC, &peLatencyAccum, &peLatencyOther, &peEnergyADC, &peEnergyAccum, &peEnergyOther);

							*readLatency = max(PEreadLatency, (*readLatency));
							*readDynamicEnergy += PEreadDynamicEnergy;
							*bufferLatency = max(PEbufferLatency, (*bufferLatency));
							*bufferDynamicEnergy += PEbufferDynamicEnergy;
							*icLatency = max(PEicLatency,(*icLatency));
							*icDynamicEnergy += PEicDynamicEnergy;

							*coreLatencyADC = MAX(peLatencyADC, (*coreLatencyADC));
							*coreLatencyAccum = MAX(peLatencyAccum, (*coreLatencyAccum));
							*coreLatencyOther = MAX(peLatencyOther, (*coreLatencyOther));

							*coreEnergyADC += peEnergyADC;
							*coreEnergyAccum += peEnergyAccum;
							*coreEnergyOther += peEnergyOther;
						}
					}
				}
				*readLatency /= (speedUpRow*speedUpCol/numPE);   // further speedup in PE level
				*coreLatencyADC /= (speedUpRow*speedUpCol/numPE);
				*coreLatencyAccum /= (speedUpRow*speedUpCol/numPE);
				*coreLatencyOther /= (speedUpRow*speedUpCol/numPE);

				// whether go through accumulation?
				if (ceil((double)weightMatrixRow/(double)peSize) > 1) {
					accumulation->CalculateLatency(param->numColMuxed, ceil((double)weightMatrixRow/(double)peSize), 0);
					accumulation->CalculatePower(param->numColMuxed, ceil((double)weightMatrixRow/(double)peSize));
					*readLatency += accumulation->readLatency;
					*readDynamicEnergy += accumulation->readDynamicEnergy;

					*coreLatencyAccum + accumulation->readLatency;
					*coreEnergyAccum + accumulation->readDynamicEnergy;
				}
			}

		} else {
			// no duplication --> tell PE to further partition the weight and grab data (redefine a few data-grab start-point)
			for (int i=0; i<ceil((double)sqrt((double)numPE)); i++) {
				for (int j=0; j<ceil((double)sqrt((double)numPE)); j++) {
					// each cycle assign to different PE
					if ( (i*peSize < weightMatrixRow) && (j*peSize < weightMatrixCol) ) {
						// assign weight and input to specific tile
						int numRowMatrix = min(peSize, (double) weightMatrixRow-i*peSize);
						int numColMatrix = min(peSize, (double) weightMatrixCol-j*peSize);

						vector<vector<double> > pEMemory;
						pEMemory = CopyPEArray(newMemory, i*peSize, j*peSize, numRowMatrix, numColMatrix);
						vector<vector<double> > pEInput;
						pEInput = CopyPEInput(inputVector, i*peSize, numInVector, numRowMatrix);

						ProcessingUnitCalculatePerformance(subArrayInPE, pEMemory, pEMemory, pEInput, 1, 1, numSubArrayRow, numSubArrayCol, numRowMatrix,
												numColMatrix, numInVector, cell, &PEreadLatency, &PEreadDynamicEnergy, &PEleakage,
												&PEbufferLatency, &PEbufferDynamicEnergy, &PEicLatency, &PEicDynamicEnergy,
												&peLatencyADC, &peLatencyAccum, &peLatencyOther, &peEnergyADC, &peEnergyAccum, &peEnergyOther);
					}
					*readLatency = max(PEreadLatency, (*readLatency));
					*readDynamicEnergy += PEreadDynamicEnergy;

					*bufferLatency = max(PEbufferLatency, (*bufferLatency));
					*bufferDynamicEnergy += PEbufferDynamicEnergy;
					*icLatency = max(PEicLatency,(*icLatency));
					*icDynamicEnergy += PEicDynamicEnergy;

					*coreLatencyADC = MAX(peLatencyADC, (*coreLatencyADC));
					*coreLatencyAccum = MAX(peLatencyAccum, (*coreLatencyAccum));
					*coreLatencyOther = MAX(peLatencyOther, (*coreLatencyOther));

					*coreEnergyADC += peEnergyADC;
					*coreEnergyAccum += peEnergyAccum;
					*coreEnergyOther += peEnergyOther;
				}
			}
			accumulation->CalculateLatency(param->numColMuxed, ceil((double)sqrt((double)numPE)), 0);
			accumulation->CalculatePower(param->numColMuxed, ceil((double)sqrt((double)numPE)));
			*readLatency += accumulation->readLatency;
			*readDynamicEnergy += accumulation->readDynamicEnergy;
			*coreLatencyAccum + accumulation->readLatency;
			*coreEnergyAccum + accumulation->readDynamicEnergy;
		}
		if (!param->chipActivation) {
			if (param->reLu) {
				reLu->CalculateLatency(param->numColMuxed);
				reLu->CalculatePower(param->numColMuxed);
				*readLatency += reLu->readLatency;
				*readDynamicEnergy += reLu->readDynamicEnergy;
				*coreLatencyOther += reLu->readLatency;
				*coreEnergyOther += reLu->readDynamicEnergy;
				outputBuffer->CalculateLatency(weightMatrixCol*(1+reLu->numBit), numInVector/param->numBitInput, weightMatrixCol*(1+reLu->numBit), numInVector/param->numBitInput);
				outputBuffer->CalculatePower(weightMatrixCol*(1+reLu->numBit), numInVector/param->numBitInput, weightMatrixCol*(1+reLu->numBit), numInVector/param->numBitInput);
			} else {
				sigmoid->CalculateLatency(param->numColMuxed);
				sigmoid->CalculatePower(param->numColMuxed);
				*readLatency += sigmoid->readLatency;
				*readDynamicEnergy += sigmoid->readDynamicEnergy;
				*coreLatencyOther += sigmoid->readLatency;
				*coreEnergyOther += sigmoid->readDynamicEnergy;
				outputBuffer->CalculateLatency(weightMatrixCol*(1+sigmoid->numYbit), numInVector/param->numBitInput, weightMatrixCol*(1+sigmoid->numYbit), numInVector/param->numBitInput);
				outputBuffer->CalculatePower(weightMatrixCol*(1+sigmoid->numYbit), numInVector/param->numBitInput, weightMatrixCol*(1+sigmoid->numYbit), numInVector/param->numBitInput);
			}
		} else {
			outputBuffer->CalculateLatency(weightMatrixCol*(1+accumulation->numAdderBit), numInVector/param->numBitInput, weightMatrixCol*(1+accumulation->numAdderBit), numInVector/param->numBitInput);
			outputBuffer->CalculatePower(weightMatrixCol*(1+accumulation->numAdderBit), numInVector/param->numBitInput, weightMatrixCol*(1+accumulation->numAdderBit), numInVector/param->numBitInput);
		}
		//considering buffer activation: no matter speedup or not, the total number of data transferred is fixed
		inputBuffer->CalculateLatency(weightMatrixRow, numInVector/param->numBitInput, weightMatrixRow, numInVector/param->numBitInput);
		inputBuffer->CalculatePower(weightMatrixRow, numInVector, weightMatrixRow, numInVector);
		*readLatency += inputBuffer->readLatency + inputBuffer->writeLatency;
		*readDynamicEnergy += inputBuffer->readDynamicEnergy + inputBuffer->writeDynamicEnergy;
		*readLatency += outputBuffer->readLatency + outputBuffer->writeLatency;
		*readDynamicEnergy += outputBuffer->readDynamicEnergy + outputBuffer->writeDynamicEnergy;
		// used to define travel distance
		double PEheight, PEwidth, PEbufferArea;
		int numSubArray = ceil((double) peSize/(double) param->numRowSubArray)*ceil((double) peSize/(double) param->numColSubArray);
		vector<double> PEarea;
		PEarea = ProcessingUnitCalculateArea(subArrayInPE, ceil((double)sqrt((double)numSubArray)), ceil((double)sqrt((double)numSubArray)), &PEheight, &PEwidth, &PEbufferArea);
		hTree->CalculateLatency(NULL, NULL, NULL, NULL, PEheight, PEwidth, weightMatrixRow*numInVector/param->numBitInput/hTree->busWidth+weightMatrixCol*numInVector/param->numBitInput/hTree->busWidth);
		hTree->CalculatePower(NULL, NULL, NULL, NULL, PEheight, PEwidth, hTree->busWidth, weightMatrixRow*numInVector/param->numBitInput/hTree->busWidth+weightMatrixCol*numInVector/param->numBitInput/hTree->busWidth);

		*readLatency += hTree->readLatency;
		*readDynamicEnergy += hTree->readDynamicEnergy;

		*bufferLatency += inputBuffer->readLatency + outputBuffer->readLatency + inputBuffer->writeLatency + outputBuffer->writeLatency;
		*icLatency += hTree->readLatency;
		*bufferDynamicEnergy += inputBuffer->readDynamicEnergy + outputBuffer->readDynamicEnergy + inputBuffer->writeDynamicEnergy + outputBuffer->writeDynamicEnergy;
		*icDynamicEnergy += hTree->readDynamicEnergy;

		//*coreLatencyOther += inputBuffer->readLatency + inputBuffer->writeLatency + outputBuffer->readLatency + outputBuffer->writeLatency + hTree->readLatency;
		//*coreEnergyOther += inputBuffer->readDynamicEnergy + inputBuffer->writeDynamicEnergy + outputBuffer->readDynamicEnergy + outputBuffer->writeDynamicEnergy + hTree->readDynamicEnergy;

	} else {  // novel Mapping
		for (int i=0; i<numPE; i++) {
			if (i*peSize < weightMatrixRow) {
				vector<vector<double> > pEMemory;
				pEMemory = CopyPEArray(newMemory, i*peSize, 0, weightMatrixRow/numPE, weightMatrixCol);
				vector<vector<double> > pEInput;
				pEInput = CopyPEInput(inputVector, i*peSize, numInVector, weightMatrixRow/numPE);

				ProcessingUnitCalculatePerformance(subArrayInPE, pEMemory, pEMemory, pEInput, 1, 1, numSubArrayRow, numSubArrayCol, weightMatrixRow/numPE,
										weightMatrixCol, numInVector, cell, &PEreadLatency, &PEreadDynamicEnergy, &PEleakage,
										&PEbufferLatency, &PEbufferDynamicEnergy, &PEicLatency, &PEicDynamicEnergy,
										&peLatencyADC, &peLatencyAccum, &peLatencyOther, &peEnergyADC, &peEnergyAccum, &peEnergyOther);
			}
			*readLatency = max(PEreadLatency, (*readLatency));
			*readDynamicEnergy += PEreadDynamicEnergy;
			*bufferLatency = max(PEbufferLatency, (*bufferLatency));
			*bufferDynamicEnergy += PEbufferDynamicEnergy;
			*icLatency = max(PEicLatency,(*icLatency));
			*icDynamicEnergy += PEicDynamicEnergy;

			*coreLatencyADC = MAX(peLatencyADC, (*coreLatencyADC));
			*coreLatencyAccum = MAX(peLatencyAccum, (*coreLatencyAccum));
			*coreLatencyOther = MAX(peLatencyOther, (*coreLatencyOther));

			*coreEnergyADC += peEnergyADC;
			*coreEnergyAccum += peEnergyAccum;
			*coreEnergyOther += peEnergyOther;
		}
		*readLatency /= speedUpRow*speedUpCol;

		*coreLatencyADC /= (speedUpRow*speedUpCol);
		*coreLatencyAccum /= (speedUpRow*speedUpCol);
		*coreLatencyOther /= (speedUpRow*speedUpCol);

		accumulation->CalculateLatency(param->numColMuxed, ceil((double)sqrt((double)numPE)), 0);
		accumulation->CalculatePower(param->numColMuxed, ceil((double)sqrt((double)numPE)));
		*readLatency += accumulation->readLatency;
		*readDynamicEnergy += accumulation->readDynamicEnergy;

		*coreLatencyAccum + accumulation->readLatency;
		*coreEnergyAccum + accumulation->readDynamicEnergy;

		//considering buffer activation: no matter speedup or not, the total number of data transferred is fixed
		inputBuffer->CalculateLatency(weightMatrixRow, numInVector/param->numBitInput/ceil((double)sqrt((double)numPE)), weightMatrixRow, numInVector/param->numBitInput/ceil((double)sqrt((double)numPE)));
		inputBuffer->CalculatePower(weightMatrixRow, numInVector/ceil((double)sqrt((double)numPE)), weightMatrixRow, numInVector/ceil((double)sqrt((double)numPE)));
		*readLatency += inputBuffer->readLatency + inputBuffer->writeLatency;
		*readDynamicEnergy += inputBuffer->readDynamicEnergy + inputBuffer->writeDynamicEnergy;

		if (!param->chipActivation) {
			if (param->reLu) {
				reLu->CalculateLatency(param->numColMuxed);
				reLu->CalculatePower(param->numColMuxed);
				*readLatency += reLu->readLatency;
				*readDynamicEnergy += reLu->readDynamicEnergy;
				*coreLatencyOther += reLu->readLatency;
				*coreEnergyOther += reLu->readDynamicEnergy;
				outputBuffer->CalculateLatency(weightMatrixCol*(1+reLu->numBit), numInVector/param->numBitInput/ceil((double)sqrt((double)numPE)),
									weightMatrixCol*(1+reLu->numBit), numInVector/param->numBitInput/ceil((double)sqrt((double)numPE)));
				outputBuffer->CalculatePower(weightMatrixCol*(1+reLu->numBit), numInVector/param->numBitInput/ceil((double)sqrt((double)numPE)),
									weightMatrixCol*(1+reLu->numBit), numInVector/param->numBitInput/ceil((double)sqrt((double)numPE)));
			} else {
				sigmoid->CalculateLatency(param->numColMuxed);
				sigmoid->CalculatePower(param->numColMuxed);
				*readLatency += sigmoid->readLatency;
				*readDynamicEnergy += sigmoid->readDynamicEnergy;
				*coreLatencyOther += sigmoid->readLatency;
				*coreEnergyOther += sigmoid->readDynamicEnergy;
				outputBuffer->CalculateLatency(weightMatrixCol*(1+sigmoid->numYbit), numInVector/param->numBitInput/ceil((double)sqrt((double)numPE)),
									weightMatrixCol*(1+sigmoid->numYbit), numInVector/param->numBitInput/ceil((double)sqrt((double)numPE)));
				outputBuffer->CalculatePower(weightMatrixCol*(1+sigmoid->numYbit), numInVector/param->numBitInput/ceil((double)sqrt((double)numPE)),
									weightMatrixCol*(1+sigmoid->numYbit), numInVector/param->numBitInput/ceil((double)sqrt((double)numPE)));
			}
		} else {
			outputBuffer->CalculateLatency(weightMatrixCol*(1+accumulation->numAdderBit), numInVector/param->numBitInput/ceil((double)sqrt((double)numPE)),
									weightMatrixCol*(1+accumulation->numAdderBit), numInVector/param->numBitInput/ceil((double)sqrt((double)numPE)));
			outputBuffer->CalculatePower(weightMatrixCol*(1+accumulation->numAdderBit), numInVector/param->numBitInput/ceil((double)sqrt((double)numPE)),
									weightMatrixCol*(1+accumulation->numAdderBit), numInVector/param->numBitInput/ceil((double)sqrt((double)numPE)));
		}

		*readLatency += outputBuffer->readLatency + outputBuffer->writeLatency;
		*readDynamicEnergy += outputBuffer->readDynamicEnergy + outputBuffer->writeDynamicEnergy;

		// used to define travel distance
		double PEheight, PEwidth, PEbufferArea;
		int numSubArray = ceil((double) peSize/(double) param->numRowSubArray)*ceil((double) peSize/(double) param->numColSubArray);
		vector<double> PEarea;
		PEarea = ProcessingUnitCalculateArea(subArrayInPE, ceil((double)sqrt((double)numSubArray)), ceil((double)sqrt((double)numSubArray)), &PEheight, &PEwidth, &PEbufferArea);
		hTree->CalculateLatency(0, 0, 1, 1, PEheight, PEwidth, (weightMatrixRow+weightMatrixCol)*numInVector/param->numBitInput/hTree->busWidth/ceil((double)sqrt((double)numPE)));
		hTree->CalculatePower(0, 0, 1, 1, PEheight, PEwidth, hTree->busWidth, (weightMatrixRow+weightMatrixCol)*numInVector/param->numBitInput/hTree->busWidth/ceil((double)sqrt((double)numPE)));

		*readLatency += hTree->readLatency;
		*readDynamicEnergy += hTree->readDynamicEnergy;

		*bufferLatency += inputBuffer->readLatency + outputBuffer->readLatency + inputBuffer->writeLatency + outputBuffer->writeLatency;
		*icLatency += hTree->readLatency;
		*bufferDynamicEnergy += inputBuffer->readDynamicEnergy + outputBuffer->readDynamicEnergy + inputBuffer->writeDynamicEnergy + outputBuffer->writeDynamicEnergy;
		*icDynamicEnergy += hTree->readDynamicEnergy;

		*coreLatencyOther += inputBuffer->readLatency + inputBuffer->writeLatency + outputBuffer->readLatency + outputBuffer->writeLatency + hTree->readLatency;
		*coreEnergyOther += inputBuffer->readDynamicEnergy + inputBuffer->writeDynamicEnergy + outputBuffer->readDynamicEnergy + outputBuffer->writeDynamicEnergy + hTree->readDynamicEnergy;
	}
	*leakage = PEleakage*numPE + accumulation->leakage + inputBuffer->leakage + outputBuffer->leakage;
}

void TileCalculatePerformance_new(const vector<vector<double> > &newMemory, const vector<vector<double> > &oldMemory, const vector<vector<double> > &inputVector, int novelMap, double numPE,
							double peSize_x, double peSize_y, int speedUpRow, int speedUpCol, int weightMatrixRow, int weightMatrixCol, int numInVector, MemCell& cell, double *readLatency, double *readDynamicEnergy, double *leakage,
							double *bufferLatency, double *bufferDynamicEnergy, double *icLatency, double *icDynamicEnergy,
							double *coreLatencyADC, double *coreLatencyAccum, double *coreLatencyOther, double *coreEnergyADC, double *coreEnergyAccum, double *coreEnergyOther) {

	/*** sweep PE ***/
	int numRowPerSynapse, numColPerSynapse;
	numRowPerSynapse = param->numRowPerSynapse;
	numColPerSynapse = param->numColPerSynapse;
	double PEreadLatency, PEreadDynamicEnergy, PEleakage, PEbufferLatency, PEbufferDynamicEnergy, PEicLatency, PEicDynamicEnergy;
	double peLatencyADC, peLatencyAccum, peLatencyOther, peEnergyADC, peEnergyAccum, peEnergyOther;
	int numSubArrayRow = ceil((double)peSize_x/(double)param->numRowSubArray);
	int numSubArrayCol = ceil((double)peSize_y/(double)param->numColSubArray);

	*readLatency = 0;
	*readDynamicEnergy = 0;
	*leakage = 0;
	*bufferLatency = 0;
	*bufferDynamicEnergy = 0;
	*icLatency = 0;
	*icDynamicEnergy = 0;
	*coreEnergyADC = 0;
	*coreEnergyAccum = 0;
	*coreEnergyOther = 0;
	*coreLatencyADC = 0;
	*coreLatencyAccum = 0;
	*coreLatencyOther = 0;

	//cout<<"\n The number of PE is: "<<numPE<<"\n"<<endl;

	if (!novelMap) {   // conventional Mapping
		if (speedUpRow*speedUpCol > 1) {
			if ( (speedUpRow >= ceil(sqrt((double)numPE))) && (speedUpCol >= ceil(sqrt((double)numPE))) ) {
				// duplication in PE or subArray --> tell each PE to take the whole assigned weight  --> "fully" duplication
				// assign weight and input to specific tile
				vector<vector<double> > pEMemory;
				pEMemory = CopyPEArray(newMemory, 0, 0, weightMatrixRow, weightMatrixCol);
				vector<vector<double> > pEInput;
				pEInput = CopyPEInput(inputVector, 0, numInVector, weightMatrixRow);

				ProcessingUnitCalculatePerformance(subArrayInPE, pEMemory, pEMemory, pEInput, ceil((double)speedUpRow/sqrt((double)numPE)), ceil((double)speedUpCol/sqrt((double)numPE)),
											numSubArrayRow, numSubArrayCol, weightMatrixRow, weightMatrixCol, numInVector, cell, &PEreadLatency, &PEreadDynamicEnergy, &PEleakage,
											&PEbufferLatency, &PEbufferDynamicEnergy, &PEicLatency, &PEicDynamicEnergy,
											&peLatencyADC, &peLatencyAccum, &peLatencyOther, &peEnergyADC, &peEnergyAccum, &peEnergyOther);

				*readLatency = PEreadLatency/numPE;  // further speed up in PE level
				*readDynamicEnergy = PEreadDynamicEnergy;   // since subArray.cpp takes all input vectors, no need to *numPE here
				*bufferLatency = PEbufferLatency/numPE;
				*bufferDynamicEnergy = PEbufferDynamicEnergy;
				*icLatency = PEicLatency/numPE;
				*icDynamicEnergy = PEicDynamicEnergy;
				//cout<<"\n The numPE is:"<<numPE<<endl;
				//cout<<"\n The peLatencyADC is:"<<peLatencyADC<<endl;
				*coreLatencyADC = peLatencyADC/numPE;
				//cout<<"\n peLatencyAccum/numPE is: "<<(peLatencyAccum/numPE)<<endl;
				*coreLatencyAccum = peLatencyAccum/numPE;
				*coreLatencyOther = peLatencyOther/numPE;

				*coreEnergyADC = peEnergyADC;
				*coreEnergyAccum = peEnergyAccum;
				*coreEnergyOther = peEnergyOther;
				// no accumulation access
			} else {
				// # duplication is smaller then # PE, means only a group of PE take the assigned weight  --> not "fully" duplication
				// also need to redefine a few data-grab start-point
				for (int i=0; i<ceil((double)weightMatrixRow/(double)peSize_x); i++) {
					for (int j=0; j<ceil((double)weightMatrixCol/(double)peSize_y); j++) {
						if ( (i*peSize_x < weightMatrixRow) && (j*peSize_y < weightMatrixCol) ) {
							int numRowMatrix = min(peSize_x, (double) weightMatrixRow-i*peSize_x);
							int numColMatrix = min(peSize_y, (double) weightMatrixCol-j*peSize_y);

							// assign weight and input to specific tile
							vector<vector<double> > pEMemory;
							pEMemory = CopyPEArray(newMemory, i*peSize_x, j*peSize_y, numRowMatrix, numColMatrix);
							vector<vector<double> > pEInput;
							pEInput = CopyPEInput(inputVector, i*peSize_x, numInVector, numRowMatrix);

							ProcessingUnitCalculatePerformance(subArrayInPE, pEMemory, pEMemory, pEInput, ceil((double)speedUpRow/sqrt((double)numPE)), ceil((double)speedUpCol/sqrt((double)numPE)),
												numSubArrayRow, numSubArrayCol, numRowMatrix, numColMatrix, numInVector, cell, &PEreadLatency, &PEreadDynamicEnergy, &PEleakage,
												&PEbufferLatency, &PEbufferDynamicEnergy, &PEicLatency, &PEicDynamicEnergy,
												&peLatencyADC, &peLatencyAccum, &peLatencyOther, &peEnergyADC, &peEnergyAccum, &peEnergyOther);
							//
							*readLatency = max(PEreadLatency, (*readLatency));
							*readDynamicEnergy += PEreadDynamicEnergy;
							*bufferLatency = max(PEbufferLatency, (*bufferLatency));
							*bufferDynamicEnergy += PEbufferDynamicEnergy;
							*icLatency = max(PEicLatency,(*icLatency));
							*icDynamicEnergy += PEicDynamicEnergy;

							*coreLatencyADC = MAX(peLatencyADC, (*coreLatencyADC));
							//cout<<"\n The numPE is:"<<numPE<<endl;
							//cout<<"\n The peLatencyADC is:"<<peLatencyADC<<endl;
							//cout<<"\n peLatencyAccum/numPE with duplication is: "<<(peLatencyAccum)<<endl;
							*coreLatencyAccum = MAX(peLatencyAccum, (*coreLatencyAccum));
							*coreLatencyOther = MAX(peLatencyOther, (*coreLatencyOther));

							*coreEnergyADC += peEnergyADC;
							*coreEnergyAccum += peEnergyAccum;
							*coreEnergyOther += peEnergyOther;
						}
					}
				}
				*readLatency /= (speedUpRow*speedUpCol/numPE);   // further speedup in PE level
				*coreLatencyADC /= (speedUpRow*speedUpCol/numPE);
				*coreLatencyAccum /= (speedUpRow*speedUpCol/numPE);
				//cout<<"\n After duplication we normalize to get *coreLatencyAccum as: "<<(*coreLatencyAccum)<<endl;
				*coreLatencyOther /= (speedUpRow*speedUpCol/numPE);

				// whether go through accumulation?
				if (ceil((double)weightMatrixRow/(double)peSize_x) > 1) {
					accumulation->CalculateLatency(param->numColMuxed, ceil((double)weightMatrixRow/(double)peSize_x), 0);
					accumulation->CalculatePower(param->numColMuxed, ceil((double)weightMatrixRow/(double)peSize_x));
					*readLatency += accumulation->readLatency;
					*readDynamicEnergy += accumulation->readDynamicEnergy;
					//cout<<"\n Accumulation read latency at tile level is: "<<accumulation->readLatency<<endl;
					*coreLatencyAccum += accumulation->readLatency;
					*coreEnergyAccum += accumulation->readDynamicEnergy;
				}
			}

		}

		else {
			// no duplication --> tell PE to further partition the weight and grab data (redefine a few data-grab start-point)
			for (int i=0; i<ceil((double)sqrt((double)numPE)); i++) {
				for (int j=0; j<ceil((double)sqrt((double)numPE)); j++) {
					// each cycle assign to different PE
					if ( (i*peSize_x < weightMatrixRow) && (j*peSize_y < weightMatrixCol) ) {     //Edited only till here ***************************************
						// assign weight and input to specific tile
						int numRowMatrix = min(peSize_x, (double) weightMatrixRow-i*peSize_x);
						int numColMatrix = min(peSize_y, (double) weightMatrixCol-j*peSize_y);

						vector<vector<double> > pEMemory;
						pEMemory = CopyPEArray(newMemory, i*peSize_x, j*peSize_y, numRowMatrix, numColMatrix);
						vector<vector<double> > pEInput;
						pEInput = CopyPEInput(inputVector, i*peSize_x, numInVector, numRowMatrix);

						ProcessingUnitCalculatePerformance(subArrayInPE, pEMemory, pEMemory, pEInput, 1, 1, numSubArrayRow, numSubArrayCol, numRowMatrix,
												numColMatrix, numInVector, cell, &PEreadLatency, &PEreadDynamicEnergy, &PEleakage,
												&PEbufferLatency, &PEbufferDynamicEnergy, &PEicLatency, &PEicDynamicEnergy,
												&peLatencyADC, &peLatencyAccum, &peLatencyOther, &peEnergyADC, &peEnergyAccum, &peEnergyOther);
					}
					*readLatency = max(PEreadLatency, (*readLatency));
					*readDynamicEnergy += PEreadDynamicEnergy;

					*bufferLatency = max(PEbufferLatency, (*bufferLatency));
					*bufferDynamicEnergy += PEbufferDynamicEnergy;
					*icLatency = max(PEicLatency,(*icLatency));
					*icDynamicEnergy += PEicDynamicEnergy;

					*coreLatencyADC = MAX(peLatencyADC, (*coreLatencyADC));
					//cout<<"\n The numPE is:"<<numPE<<endl;
					//cout<<"\n The peLatencyADC is:"<<peLatencyADC<<endl;
					*coreLatencyAccum = MAX(peLatencyAccum, (*coreLatencyAccum));
					*coreLatencyOther = MAX(peLatencyOther, (*coreLatencyOther));
					//
					*coreEnergyADC += peEnergyADC;
					*coreEnergyAccum += peEnergyAccum;
					*coreEnergyOther += peEnergyOther;
				}
			}
			accumulation->CalculateLatency(param->numColMuxed, ceil((double)sqrt((double)numPE)), 0);
			accumulation->CalculatePower(param->numColMuxed, ceil((double)sqrt((double)numPE)));
			*readLatency += accumulation->readLatency;
			*readDynamicEnergy += accumulation->readDynamicEnergy;
			*coreLatencyAccum += accumulation->readLatency;
			//cout<<"\n *coreLatencyAccum read latency at tile level wihtout duplication is: "<<*coreLatencyAccum<<endl;
			*coreEnergyAccum += accumulation->readDynamicEnergy;
		}
		if (!param->chipActivation) {
			if (param->reLu) {
				reLu->CalculateLatency(param->numColMuxed);
				reLu->CalculatePower(param->numColMuxed);
				*readLatency += reLu->readLatency;
				*readDynamicEnergy += reLu->readDynamicEnergy;
				*coreLatencyOther += reLu->readLatency;
				*coreEnergyOther += reLu->readDynamicEnergy;
				outputBuffer->CalculateLatency(weightMatrixCol*(1+reLu->numBit), numInVector/param->numBitInput, weightMatrixCol*(1+reLu->numBit), numInVector/param->numBitInput);
				outputBuffer->CalculatePower(weightMatrixCol*(1+reLu->numBit), numInVector/param->numBitInput, weightMatrixCol*(1+reLu->numBit), numInVector/param->numBitInput);
			} else {
				sigmoid->CalculateLatency(param->numColMuxed);
				sigmoid->CalculatePower(param->numColMuxed);
				*readLatency += sigmoid->readLatency;
				*readDynamicEnergy += sigmoid->readDynamicEnergy;
				*coreLatencyOther += sigmoid->readLatency;
				*coreEnergyOther += sigmoid->readDynamicEnergy;
				outputBuffer->CalculateLatency(weightMatrixCol*(1+sigmoid->numYbit), numInVector/param->numBitInput, weightMatrixCol*(1+sigmoid->numYbit), numInVector/param->numBitInput);
				outputBuffer->CalculatePower(weightMatrixCol*(1+sigmoid->numYbit), numInVector/param->numBitInput, weightMatrixCol*(1+sigmoid->numYbit), numInVector/param->numBitInput);
			}
		} else {
			outputBuffer->CalculateLatency(weightMatrixCol*(1+accumulation->numAdderBit), numInVector/param->numBitInput, weightMatrixCol*(1+accumulation->numAdderBit), numInVector/param->numBitInput);
			outputBuffer->CalculatePower(weightMatrixCol*(1+accumulation->numAdderBit), numInVector/param->numBitInput, weightMatrixCol*(1+accumulation->numAdderBit), numInVector/param->numBitInput);
		}
		//considering buffer activation: no matter speedup or not, the total number of data transferred is fixed
		inputBuffer->CalculateLatency(weightMatrixRow, numInVector/param->numBitInput, weightMatrixRow, numInVector/param->numBitInput);
		inputBuffer->CalculatePower(weightMatrixRow, numInVector, weightMatrixRow, numInVector);
		*readLatency += inputBuffer->readLatency + inputBuffer->writeLatency;
		*readDynamicEnergy += inputBuffer->readDynamicEnergy + inputBuffer->writeDynamicEnergy;
		*readLatency += outputBuffer->readLatency + outputBuffer->writeLatency;
		*readDynamicEnergy += outputBuffer->readDynamicEnergy + outputBuffer->writeDynamicEnergy;
		// used to define travel distance
		double PEheight, PEwidth, PEbufferArea;
		int numSubArray = ceil((double) peSize_x/(double) param->numRowSubArray)*ceil((double) peSize_y/(double) param->numColSubArray);
		vector<double> PEarea;
		PEarea = ProcessingUnitCalculateArea(subArrayInPE, ceil((double)sqrt((double)numSubArray)), ceil((double)sqrt((double)numSubArray)), &PEheight, &PEwidth, &PEbufferArea);
		hTree->CalculateLatency(NULL, NULL, NULL, NULL, PEheight, PEwidth, weightMatrixRow*numInVector/param->numBitInput/hTree->busWidth+weightMatrixCol*numInVector/param->numBitInput/hTree->busWidth);
		hTree->CalculatePower(NULL, NULL, NULL, NULL, PEheight, PEwidth, hTree->busWidth, weightMatrixRow*numInVector/param->numBitInput/hTree->busWidth+weightMatrixCol*numInVector/param->numBitInput/hTree->busWidth);

		*readLatency += hTree->readLatency;
		*readDynamicEnergy += hTree->readDynamicEnergy;

		*bufferLatency += inputBuffer->readLatency + outputBuffer->readLatency + inputBuffer->writeLatency + outputBuffer->writeLatency;
		*icLatency += hTree->readLatency;
		*bufferDynamicEnergy += inputBuffer->readDynamicEnergy + outputBuffer->readDynamicEnergy + inputBuffer->writeDynamicEnergy + outputBuffer->writeDynamicEnergy;
		*icDynamicEnergy += hTree->readDynamicEnergy;
		//
		//*coreLatencyOther += inputBuffer->readLatency + inputBuffer->writeLatency + outputBuffer->readLatency + outputBuffer->writeLatency + hTree->readLatency;
		//cout<<"\n coreLatencyOther after addition of other components are is: "<<*coreLatencyOther<<"\n"<<endl;
		//*coreEnergyOther += inputBuffer->readDynamicEnergy + inputBuffer->writeDynamicEnergy + outputBuffer->readDynamicEnergy + outputBuffer->writeDynamicEnergy + hTree->readDynamicEnergy;

	} else {  // novel Mapping
		for (int i=0; i<numPE; i++) {
			if (i*peSize_x < weightMatrixRow) {
				vector<vector<double> > pEMemory;
				pEMemory = CopyPEArray(newMemory, i*peSize_x, 0, weightMatrixRow/numPE, weightMatrixCol);
				vector<vector<double> > pEInput;
				pEInput = CopyPEInput(inputVector, i*peSize_x, numInVector, weightMatrixRow/numPE);

				ProcessingUnitCalculatePerformance(subArrayInPE, pEMemory, pEMemory, pEInput, 1, 1, numSubArrayRow, numSubArrayCol, weightMatrixRow/numPE,
										weightMatrixCol, numInVector, cell, &PEreadLatency, &PEreadDynamicEnergy, &PEleakage,
										&PEbufferLatency, &PEbufferDynamicEnergy, &PEicLatency, &PEicDynamicEnergy,
										&peLatencyADC, &peLatencyAccum, &peLatencyOther, &peEnergyADC, &peEnergyAccum, &peEnergyOther);
			}
			*readLatency = max(PEreadLatency, (*readLatency));
			*readDynamicEnergy += PEreadDynamicEnergy;
			*bufferLatency = max(PEbufferLatency, (*bufferLatency));
			*bufferDynamicEnergy += PEbufferDynamicEnergy;
			*icLatency = max(PEicLatency,(*icLatency));
			*icDynamicEnergy += PEicDynamicEnergy;

			*coreLatencyADC = MAX(peLatencyADC, (*coreLatencyADC));
			*coreLatencyAccum = MAX(peLatencyAccum, (*coreLatencyAccum));
			*coreLatencyOther = MAX(peLatencyOther, (*coreLatencyOther));

			*coreEnergyADC += peEnergyADC;
			*coreEnergyAccum += peEnergyAccum;
			*coreEnergyOther += peEnergyOther;
		}
		*readLatency /= speedUpRow*speedUpCol;

		*coreLatencyADC /= (speedUpRow*speedUpCol);
		*coreLatencyAccum /= (speedUpRow*speedUpCol);
		*coreLatencyOther /= (speedUpRow*speedUpCol);

		accumulation->CalculateLatency(param->numColMuxed, ceil((double)sqrt((double)numPE)), 0);
		accumulation->CalculatePower(param->numColMuxed, ceil((double)sqrt((double)numPE)));
		*readLatency += accumulation->readLatency;
		*readDynamicEnergy += accumulation->readDynamicEnergy;

		*coreLatencyAccum += accumulation->readLatency;
		*coreEnergyAccum += accumulation->readDynamicEnergy;

		//considering buffer activation: no matter speedup or not, the total number of data transferred is fixed
		inputBuffer->CalculateLatency(weightMatrixRow, numInVector/param->numBitInput/ceil((double)sqrt((double)numPE)), weightMatrixRow, numInVector/param->numBitInput/ceil((double)sqrt((double)numPE)));
		inputBuffer->CalculatePower(weightMatrixRow, numInVector/ceil((double)sqrt((double)numPE)), weightMatrixRow, numInVector/ceil((double)sqrt((double)numPE)));
		*readLatency += inputBuffer->readLatency + inputBuffer->writeLatency;
		*readDynamicEnergy += inputBuffer->readDynamicEnergy + inputBuffer->writeDynamicEnergy;

		if (!param->chipActivation) {
			if (param->reLu) {
				reLu->CalculateLatency(param->numColMuxed);
				reLu->CalculatePower(param->numColMuxed);
				*readLatency += reLu->readLatency;
				*readDynamicEnergy += reLu->readDynamicEnergy;
				*coreLatencyOther += reLu->readLatency;
				*coreEnergyOther += reLu->readDynamicEnergy;
				outputBuffer->CalculateLatency(weightMatrixCol*(1+reLu->numBit), numInVector/param->numBitInput/ceil((double)sqrt((double)numPE)),
									weightMatrixCol*(1+reLu->numBit), numInVector/param->numBitInput/ceil((double)sqrt((double)numPE)));
				outputBuffer->CalculatePower(weightMatrixCol*(1+reLu->numBit), numInVector/param->numBitInput/ceil((double)sqrt((double)numPE)),
									weightMatrixCol*(1+reLu->numBit), numInVector/param->numBitInput/ceil((double)sqrt((double)numPE)));
			} else {
				sigmoid->CalculateLatency(param->numColMuxed);
				sigmoid->CalculatePower(param->numColMuxed);
				*readLatency += sigmoid->readLatency;
				*readDynamicEnergy += sigmoid->readDynamicEnergy;
				*coreLatencyOther += sigmoid->readLatency;
				*coreEnergyOther += sigmoid->readDynamicEnergy;
				outputBuffer->CalculateLatency(weightMatrixCol*(1+sigmoid->numYbit), numInVector/param->numBitInput/ceil((double)sqrt((double)numPE)),
									weightMatrixCol*(1+sigmoid->numYbit), numInVector/param->numBitInput/ceil((double)sqrt((double)numPE)));
				outputBuffer->CalculatePower(weightMatrixCol*(1+sigmoid->numYbit), numInVector/param->numBitInput/ceil((double)sqrt((double)numPE)),
									weightMatrixCol*(1+sigmoid->numYbit), numInVector/param->numBitInput/ceil((double)sqrt((double)numPE)));
			}
		} else {
			outputBuffer->CalculateLatency(weightMatrixCol*(1+accumulation->numAdderBit), numInVector/param->numBitInput/ceil((double)sqrt((double)numPE)),
									weightMatrixCol*(1+accumulation->numAdderBit), numInVector/param->numBitInput/ceil((double)sqrt((double)numPE)));
			outputBuffer->CalculatePower(weightMatrixCol*(1+accumulation->numAdderBit), numInVector/param->numBitInput/ceil((double)sqrt((double)numPE)),
									weightMatrixCol*(1+accumulation->numAdderBit), numInVector/param->numBitInput/ceil((double)sqrt((double)numPE)));
		}

		*readLatency += outputBuffer->readLatency + outputBuffer->writeLatency;
		*readDynamicEnergy += outputBuffer->readDynamicEnergy + outputBuffer->writeDynamicEnergy;

		// used to define travel distance
		double PEheight, PEwidth, PEbufferArea;
		int numSubArray = ceil((double) peSize_x/(double) param->numRowSubArray)*ceil((double) peSize_y/(double) param->numColSubArray);
		vector<double> PEarea;
		PEarea = ProcessingUnitCalculateArea(subArrayInPE, ceil((double)sqrt((double)numSubArray)), ceil((double)sqrt((double)numSubArray)), &PEheight, &PEwidth, &PEbufferArea);
		hTree->CalculateLatency(0, 0, 1, 1, PEheight, PEwidth, (weightMatrixRow+weightMatrixCol)*numInVector/param->numBitInput/hTree->busWidth/ceil((double)sqrt((double)numPE)));
		hTree->CalculatePower(0, 0, 1, 1, PEheight, PEwidth, hTree->busWidth, (weightMatrixRow+weightMatrixCol)*numInVector/param->numBitInput/hTree->busWidth/ceil((double)sqrt((double)numPE)));

		*readLatency += hTree->readLatency;
		*readDynamicEnergy += hTree->readDynamicEnergy;

		*bufferLatency += inputBuffer->readLatency + outputBuffer->readLatency + inputBuffer->writeLatency + outputBuffer->writeLatency;
		*icLatency += hTree->readLatency;
		*bufferDynamicEnergy += inputBuffer->readDynamicEnergy + outputBuffer->readDynamicEnergy + inputBuffer->writeDynamicEnergy + outputBuffer->writeDynamicEnergy;
		*icDynamicEnergy += hTree->readDynamicEnergy;

		*coreLatencyOther += inputBuffer->readLatency + inputBuffer->writeLatency + outputBuffer->readLatency + outputBuffer->writeLatency + hTree->readLatency;
		*coreEnergyOther += inputBuffer->readDynamicEnergy + inputBuffer->writeDynamicEnergy + outputBuffer->readDynamicEnergy + outputBuffer->writeDynamicEnergy + hTree->readDynamicEnergy;
	}
	*leakage = PEleakage*numPE + accumulation->leakage + inputBuffer->leakage + outputBuffer->leakage;
}


vector<vector<double> > CopyPEArray(const vector<vector<double> > &orginal, int positionRow, int positionCol, int numRow, int numCol) {
	vector<vector<double> > copy;
	for (int i=0; i<numRow; i++) {
		vector<double> copyRow;
		for (int j=0; j<numCol; j++) {
			copyRow.push_back(orginal[positionRow+i][positionCol+j]);
		}
		copy.push_back(copyRow);
		copyRow.clear();
	}
	return copy;
	copy.clear();
}


vector<vector<double> > CopyPEInput(const vector<vector<double> > &orginal, int positionRow, int numInputVector, int numRow) {
	vector<vector<double> > copy;
	for (int i=0; i<numRow; i++) {
		vector<double> copyRow;
		for (int j=0; j<numInputVector; j++) {
			copyRow.push_back(orginal[positionRow+i][j]);
		}
		copy.push_back(copyRow);
		copyRow.clear();
	}
	return copy;
	copy.clear();
}
