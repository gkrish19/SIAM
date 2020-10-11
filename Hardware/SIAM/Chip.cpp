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
#include "MaxPooling.h"
#include "Sigmoid.h"
#include "BitShifter.h"
#include "AdderTree.h"
#include "Buffer.h"
#include "HTree.h"
#include "ProcessingUnit.h"
#include "Tile.h"
#include "constant.h"
#include "formula.h"
#include "Param.h"
#include "Chip.h"

using namespace std;

extern Param *param;

/*** Circuit Modules ***/
Buffer *globalBuffer;
HTree *GhTree;
AdderTree *Gaccumulation;
Sigmoid *Gsigmoid;
BitShifter *GreLu;
MaxPooling *maxPool;


vector<int> ChipDesignInitialize(InputParameter& inputParameter, Technology& tech, MemCell& cell, const vector<vector<double> > &netStructure,
					double *maxPESizeNM, double *maxTileSizeCM_x, double *maxTileSizeCM_y ,double *numPENM){

	globalBuffer = new Buffer(inputParameter, tech, cell);
	GhTree = new HTree(inputParameter, tech, cell);
	Gaccumulation = new AdderTree(inputParameter, tech, cell);
	Gsigmoid = new Sigmoid(inputParameter, tech, cell);
	GreLu = new BitShifter(inputParameter, tech, cell);
	maxPool = new MaxPooling(inputParameter, tech, cell);

	int numRowPerSynapse, numColPerSynapse;
	numRowPerSynapse = param->numRowPerSynapse;
	numColPerSynapse = param->numColPerSynapse;

	double numLayer, minCube;

	// get information of network structure
	numLayer = netStructure.size();

	*maxPESizeNM = 0;
	*maxTileSizeCM_x = 0;
	*maxTileSizeCM_y = 0;
	*numPENM = 0;

	double maxTileSizeCM;
	maxTileSizeCM = 0;

	vector<int> markNM;
	if (param->novelMapping) {
		// define number of PE in COV layers
		int most = 0;
		int numPE = 0;
		for (int i=0; i<numLayer; i++) {
			int temp = netStructure[i][3]*netStructure[i][4];
			int count = 1;
			for (int j=0; j<numLayer; j++) {
				if (temp == netStructure[j][3]*netStructure[j][4]) {
					count ++;
				}
				if (most < count) {
					most = count;
					numPE = temp;
				}
			}
		}
		*numPENM = numPE;
		// mark the layers that use novel mapping
		for (int i=0; i<numLayer; i++) {

			if ((netStructure[i][3]*netStructure[i][4]== (*numPENM))
				// large Cov layers use novel mapping
				&&(netStructure[i][2]*netStructure[i][3]*netStructure[i][4]*numRowPerSynapse >= param->numRowSubArray)) {
				markNM.push_back(1);
				minCube = pow(2, ceil((double) log2((double) netStructure[i][5]*(double) numColPerSynapse) ) );
				*maxPESizeNM = max(minCube, (*maxPESizeNM));
			} else {
				// small Cov layers and FC layers use conventional mapping
				markNM.push_back(0);
				minCube = pow(2, ceil((double) log2((double) netStructure[i][5]*(double) numColPerSynapse) ) );
				maxTileSizeCM = max(minCube, (maxTileSizeCM));
			}
		}
	} else {
		// all layers use conventional mapping
		*maxTileSizeCM_x = param->row_multiplier*param->numRowSubArray;
		*maxTileSizeCM_y = param->col_multiplier*param->numColSubArray;

		for (int i=0; i<numLayer; i++) {
			markNM.push_back(0);}
			/*
			cout<<"Net structure is: "<< (double) netStructure[i][5]<<"\n"<<endl;
			//***********Here we should change this as it assumes 16 pe per tile****************************
			minCube = pow(2, ceil((double) log2((double) netStructure[i][5]*(double) numColPerSynapse)) );
			cout<<"The minCube is: "<< minCube<<"\n"<<endl;
			*maxTileSizeCM = max(minCube, (*maxTileSizeCM_y));
			cout<<"The *maxTileSizeCM is: "<<*maxTileSizeCM<<"\n"<<endl;
		//*maxTileSizeCM =
*/
			//**********************************************************************************************
		}


	return markNM;
}


vector<vector<double> > ChipFloorPlan(bool findNumTile, bool findUtilization, bool findSpeedUp, const vector<vector<double> > &netStructure, const vector<int > &markNM,
					double maxPESizeNM, double maxTileSizeCM_x, double maxTileSizeCM_y, double numPENM,
					double *desiredNumTileNM, double *desiredPESizeNM, double *desiredNumTileCM, double *desiredTileSizeCM_x, double *desiredTileSizeCM_y, double *desiredPESizeCM_x, double *desiredPESizeCM_y, int *numTileRow, int *numTileCol) {


	int numRowPerSynapse, numColPerSynapse;
	numRowPerSynapse = param->numRowPerSynapse;
	numColPerSynapse = param->numColPerSynapse;

	double maxUtilizationNM = 0;
	double maxUtilizationCM = 0;

	vector<vector<double> > peDup;
	vector<vector<double> > subArrayDup;
	vector<vector<double> > numTileEachLayer;
	vector<vector<double> > utilizationEachLayer;
	vector<vector<double> > speedUpEachLayer;

	*desiredNumTileNM = 0;
	*desiredPESizeNM = 0;
	*desiredNumTileCM = 0;
	*desiredTileSizeCM_x = 0;
	*desiredTileSizeCM_y = 0;
	*desiredPESizeCM_x = 0;
	*desiredPESizeCM_y = 0;
	*numTileRow = 0;
	*numTileCol = 0;
	double maxTileSizeCM, *desiredTileSizeCM, *desiredPESizeCM;
	maxTileSizeCM =0;
	desiredTileSizeCM = 0;
	desiredPESizeCM = 0;


	if (param->novelMapping) {		// Novel Mappingr
		if (maxPESizeNM < 2*param->numRowSubArray) {
			cout << "ERROR: SubArray Size is too large, which break the chip hierarchey, please decrease the SubArray size! " << endl;
		}else{

			/*** Tile Design ***/
			*desiredPESizeNM = MAX(maxPESizeNM, 2*param->numRowSubArray);
			vector<double> initialDesignNM;
			initialDesignNM = TileDesignNM((*desiredPESizeNM), markNM, netStructure, numRowPerSynapse, numColPerSynapse, numPENM);
			*desiredNumTileNM = initialDesignNM[0];
			for (double thisPESize = MAX(maxPESizeNM, 2*param->numRowSubArray); thisPESize> 2*param->numRowSubArray; thisPESize/=2) {
				// for layers use novel mapping
				double thisUtilization = 0;
				vector<double> thisDesign;
				thisDesign = TileDesignNM(thisPESize, markNM, netStructure, numRowPerSynapse, numColPerSynapse, numPENM);
				thisUtilization = thisDesign[2];
				if (thisUtilization > maxUtilizationNM) {
					maxUtilizationNM = thisUtilization;
					*desiredPESizeNM = thisPESize;
					*desiredNumTileNM = thisDesign[0];
				}
			}
			*desiredTileSizeCM = MAX(maxTileSizeCM, 4*param->numRowSubArray);
			vector<double> initialDesignCM;
			initialDesignCM = TileDesignCM((*desiredTileSizeCM), markNM, netStructure, numRowPerSynapse, numColPerSynapse);
			*desiredNumTileCM = initialDesignCM[0];
			for (double thisTileSize = MAX(maxTileSizeCM, 4*param->numRowSubArray); thisTileSize > 4*param->numRowSubArray; thisTileSize/=2) {
				// for layers use conventional mapping
				double thisUtilization = 0;
				vector<double> thisDesign;
				thisDesign = TileDesignCM(thisTileSize, markNM, netStructure, numRowPerSynapse, numColPerSynapse);
				thisUtilization = thisDesign[2];
				if (thisUtilization > maxUtilizationCM) {
					maxUtilizationCM = thisUtilization;
					*desiredTileSizeCM = thisTileSize;
					*desiredNumTileCM = thisDesign[0];
				}
			}
			*desiredPESizeCM = (*desiredTileSizeCM)/2;
			/*** PE Design ***/
			for (double thisPESize = (*desiredTileSizeCM)/2; thisPESize > 2*param->numRowSubArray; thisPESize/=2) {
				// define PE Size for layers use conventional mapping
				double thisUtilization = 0;
				vector<vector<double> > thisDesign;
				thisDesign = PEDesign(true, thisPESize, (*desiredTileSizeCM), (*desiredNumTileCM), markNM, netStructure, numRowPerSynapse, numColPerSynapse);
				thisUtilization = thisDesign[1][0];
				if (thisUtilization > maxUtilizationCM) {
					maxUtilizationCM = thisUtilization;
					*desiredPESizeCM = thisPESize;
				}
			}
			peDup = PEDesign(false, (*desiredPESizeCM), (*desiredTileSizeCM), (*desiredNumTileCM), markNM, netStructure, numRowPerSynapse, numColPerSynapse);
			/*** SubArray Duplication ***/
			subArrayDup = SubArrayDup((*desiredPESizeCM), (*desiredPESizeNM), markNM, netStructure, numRowPerSynapse, numColPerSynapse);
			/*** Design SubArray ***/
			numTileEachLayer = OverallEachLayer(false, false, peDup, subArrayDup, (*desiredTileSizeCM), (*desiredPESizeNM), markNM, netStructure, numRowPerSynapse, numColPerSynapse, numPENM);
			utilizationEachLayer = OverallEachLayer(true, false, peDup, subArrayDup, (*desiredTileSizeCM), (*desiredPESizeNM), markNM, netStructure, numRowPerSynapse, numColPerSynapse, numPENM);
			speedUpEachLayer = OverallEachLayer(false, true, peDup, subArrayDup, (*desiredTileSizeCM), (*desiredPESizeNM), markNM, netStructure, numRowPerSynapse, numColPerSynapse, numPENM);
		}
	}

	else {   // all Conventional Mapping
		if ((maxTileSizeCM_x >param->row_multiplier*param->numRowSubArray) & (maxTileSizeCM_y >param->col_multiplier*param->numRowSubArray)) {
			cout << "ERROR: SubArray Size is too large, which break the chip hierarchey, please decrease the SubArray size! " << endl;
		} else {
			/*** Tile Design ***/
			*desiredTileSizeCM_y = param->row_multiplier*param->numRowSubArray;
			*desiredTileSizeCM_x = param->col_multiplier*param->numRowSubArray;
			//cout<<"The desiredTileSizeCM_x size is :"<<*desiredTileSizeCM_x<<"\n"<<endl;
			//cout<<"The desiredTileSizeCM_y size is :"<<*desiredTileSizeCM_y<<"\n"<<endl;
			vector<double> initialDesign;
			initialDesign = TileDesignCM_new((*desiredTileSizeCM_x), (*desiredTileSizeCM_y), markNM, netStructure, numRowPerSynapse, numColPerSynapse);
			*desiredNumTileCM = initialDesign[0];
			double desiredUtilCM;
			desiredUtilCM = initialDesign[2];
			//cout<<"The desired number of tiles in the whole design duirng initialization is: "<<*desiredNumTileCM<<"\n"<<endl;
			//cout<<"The utilization of the whole design duirng initialization is: "<<desiredUtilCM<<"\n"<<endl;
			/*******************************************************************************************************
			for (double thisTileSize = MAX(maxTileSizeCM, 4*param->numRowSubArray); thisTileSize > 4*param->numRowSubArray; thisTileSize/=2) {
				// for layers use conventional mapping
				double thisUtilization = 0;
				vector<double> thisDesign;
				thisDesign = TileDesignCM(thisTileSize, markNM, netStructure, numRowPerSynapse, numColPerSynapse);
				thisUtilization = thisDesign[2];
				if (thisUtilization > maxUtilizationCM) {
					maxUtilizationCM = thisUtilization;
					*desiredTileSizeCM = thisTileSize;
					*desiredNumTileCM = thisDesign[0];
				}
			}
			*/
			//*******************************************************************************************************
			//*desiredPESizeCM = (*desiredTileSizeCM)/2;
			//cout<<"The PE size before optimization is :"<<*desiredPESizeCM<<"\n"<<endl;
			*desiredPESizeCM_x = (*desiredTileSizeCM_x)/2;
			*desiredPESizeCM_y = (*desiredTileSizeCM_y)/2;
			//cout<<"The PE size x before optimization is :"<<*desiredPESizeCM_x<<"\n"<<endl;
			//cout<<"The PE size y before optimization is :"<<*desiredPESizeCM_y<<"\n"<<endl;
			/*** PE Design ***/
			double thisUtilization = 0;
			//for (double thisPESize_x = (*desiredPESizeCM_x)/2, thisPESize_y = (*desiredPESizeCM_y)/2; thisPESize_x >= 2*param->numRowSubArray, thisPESize_y >= 1*param->numRowSubArray; thisPESize_x/=2, thisPESize_y/=2) {
				// define PE Size for layers use conventional mapping

				vector<vector<double> > thisDesign;
				thisDesign = PEDesign_new(true, *desiredPESizeCM_x, *desiredPESizeCM_y, (*desiredTileSizeCM_x), (*desiredTileSizeCM_y), (*desiredNumTileCM), markNM, netStructure, numRowPerSynapse, numColPerSynapse);
				//thisDesign = PEDesign(true, thisPESize_x, (*desiredTileSizeCM_x), (*desiredNumTileCM), markNM, netStructure, numRowPerSynapse, numColPerSynapse);
				thisUtilization = thisDesign[1][0];
				maxUtilizationCM = thisUtilization;
				//if (thisUtilization > maxUtilizationCM)
				//{
				//	maxUtilizationCM = thisUtilization;
				//	*desiredPESizeCM_x = thisPESize_x;
				//	*desiredPESizeCM_y = thisPESize_y;
				//}
			//}
			//cout << "CE Size used is" << *desiredPESizeCM_x << "x" << *desiredPESizeCM_y << endl;
			//cout<<"The PE utilization is :"<<thisUtilization<<"\n"<<endl;

			peDup = PEDesign_new(false, (*desiredPESizeCM_x), (*desiredPESizeCM_y), (*desiredTileSizeCM_x), (*desiredTileSizeCM_y), (*desiredNumTileCM), markNM, netStructure, numRowPerSynapse, numColPerSynapse);
			/*** SubArray Duplication ***/
			subArrayDup = SubArrayDup_new((*desiredPESizeCM_x), (*desiredPESizeCM_y), 0, markNM, netStructure, numRowPerSynapse, numColPerSynapse);
			/*** Design SubArray ***/
			numTileEachLayer = OverallEachLayer_new(false, false, peDup, subArrayDup, (*desiredTileSizeCM_x), (*desiredTileSizeCM_y), 0, markNM, netStructure, numRowPerSynapse, numColPerSynapse, numPENM);
			utilizationEachLayer = OverallEachLayer_new(true, false, peDup, subArrayDup, (*desiredTileSizeCM_x), (*desiredTileSizeCM_y), 0, markNM, netStructure, numRowPerSynapse, numColPerSynapse, numPENM);
			speedUpEachLayer = OverallEachLayer_new(false, true, peDup, subArrayDup, (*desiredTileSizeCM_x), (*desiredTileSizeCM_y), 0, markNM, netStructure, numRowPerSynapse, numColPerSynapse, numPENM);
		}
	}

	*numTileRow = ceil((double)sqrt((double)(*desiredNumTileCM)+(double)(*desiredNumTileNM)));
	*numTileCol = ceil((double)((*desiredNumTileCM)+(*desiredNumTileNM))/(double)(*numTileRow));

	vector<vector<double> > tileLocaEachLayer;
	vector<double> tileLocaEachLayerRow;
	vector<double> tileLocaEachLayerCol;
	double thisTileTotal;
	for (int i=0; i<netStructure.size(); i++) {
		if (i==0) {
			tileLocaEachLayerRow.push_back(0);
			tileLocaEachLayerCol.push_back(0);
		} else {
			thisTileTotal += numTileEachLayer[0][i]*numTileEachLayer[1][i];
			tileLocaEachLayerRow.push_back((int)thisTileTotal/(*numTileRow));
			tileLocaEachLayerCol.push_back((int)thisTileTotal%(*numTileRow)-1);
		}
	}
	tileLocaEachLayer.push_back(tileLocaEachLayerRow);
	tileLocaEachLayer.push_back(tileLocaEachLayerCol);

	if (findNumTile) {
		return numTileEachLayer;
	} else if (findUtilization) {
		return utilizationEachLayer;
	} else if (findSpeedUp) {
		return speedUpEachLayer;
	} else {
		return tileLocaEachLayer;
	}
	peDup.clear();
	subArrayDup.clear();
	numTileEachLayer.clear();
	utilizationEachLayer.clear();
	speedUpEachLayer.clear();
}


void ChipInitialize(InputParameter& inputParameter, Technology& tech, MemCell& cell, const vector<vector<double> > &netStructure, const vector<int > &markNM, const vector<vector<double> > &numTileEachLayer,
					double numPENM, double desiredNumTileNM, double desiredPESizeNM, double desiredNumTileCM, double desiredTileSizeCM_x, double desiredTileSizeCM_y, double desiredPESizeCM_x, double desiredPESizeCM_y, int numTileRow, int numTileCol) {

	/*** Initialize Tile ***/
	double desiredTileSizeCM, desiredPESizeCM;
	if (param->novelMapping) {
		TileInitialize(inputParameter, tech, cell, numPENM, desiredPESizeNM);
	}
	int _numPE = ceil((double)(desiredTileSizeCM_x)/(double)(desiredPESizeCM_x)) + ceil((double)(desiredTileSizeCM_y)/(double)(desiredPESizeCM_y));
	TileInitialize_new(inputParameter, tech, cell, _numPE, desiredPESizeCM_x, desiredPESizeCM_y);
	// find max layer and define the global buffer: enough to hold the max layer inputs
	double maxLayerInput = 0;
	// define main global bus width
	double globalBusWidth = 0;
	// find max # tiles needed to be added at the same time
	double maxTileAdded = 0;
	for (int i=0; i<netStructure.size(); i++) {
		double input = netStructure[i][0]*netStructure[i][1]*netStructure[i][2];  // IFM_Row * IFM_Column * IFM_depth
		if (input > maxLayerInput) {
			maxLayerInput = input;
		}

		if (markNM[i] == 0) {
			globalBusWidth += (desiredTileSizeCM_y)/param->numColMuxed;
		} else {
			globalBusWidth += (desiredPESizeNM)*ceil((double)sqrt(numPENM))+(desiredPESizeNM)*ceil((double)sqrt(numPENM))/param->numColMuxed;
		}

		if (numTileEachLayer[0][i] > maxTileAdded) {
			maxTileAdded = numTileEachLayer[0][i];
			//cout<<"\n The size of the accum addition support is:"<<maxTileAdded <<endl;
		}
	}



	// have to limit the global bus width --> cannot grow dramatically with num of tile
	while (globalBusWidth > param->maxGlobalBusWidth) {
		globalBusWidth /= 2;
	}
	int pass_to_maxpool = max(desiredTileSizeCM_x, desiredTileSizeCM_y);
	globalBuffer->Initialize(param->numBitInput*maxLayerInput, ceil((double)sqrt(param->numBitInput*maxLayerInput)), 1, param->unitLengthWireResistance, param->clkFreq, param->globalBufferType);
	maxPool->Initialize(param->numBitInput, 2*2, (pass_to_maxpool));
	GhTree->Initialize((numTileRow), (numTileCol), param->globalBusDelayTolerance, globalBusWidth);

	//activation inside Tile or outside?
	if (param->chipActivation) {
		if (param->novelMapping) {
			int maxThroughputTile = (int) max((desiredTileSizeCM), ceil((double)sqrt(numPENM))*(desiredPESizeNM));
			int maxAddFromSubArray = (int) max(ceil((double)(desiredPESizeCM)/(double)param->numRowSubArray), ceil((double)(desiredPESizeNM)/(double)param->numRowSubArray));   // from subArray to ProcessingUnit
			maxAddFromSubArray *= (int) max(ceil((double)(desiredTileSizeCM)/(double)(desiredPESizeCM)), ceil((double)sqrt(numPENM)));    // from ProcessingUnit to Tile
			if (param->parallelRead) {
				Gaccumulation->Initialize((int) maxTileAdded, ceil((double) log2((double) param->levelOutput))+param->numBitInput+1+ceil((double) log2((double) maxAddFromSubArray)),
										ceil((double) maxThroughputTile/(double) param->numColMuxed));
			} else {
				Gaccumulation->Initialize((int) maxTileAdded, ceil((double) log2((double) param->numRowSubArray)+(double) param->cellBit-1)+param->numBitInput+1+ceil((double) log2((double) maxAddFromSubArray)),
										ceil((double) maxThroughputTile/(double) param->numColMuxed));
			}
			if (param->reLu) {
				GreLu->Initialize(ceil((double) maxThroughputTile/(double) param->numColMuxed), param->numBitInput, param->clkFreq);
			} else {
				Gsigmoid->Initialize(false, param->numBitInput, ceil((double) log2((double) param->numRowSubArray)+(double) param->cellBit-1)+param->numBitInput+1+log2((double) maxAddFromSubArray)+ceil((double) log2((double) maxTileAdded)),
										ceil((double) maxThroughputTile/(double) param->numColMuxed), param->clkFreq);
			}
		}

		else {
			int maxAddFromSubArray = (int) ceil((double)(desiredPESizeCM_x)/(double)param->numRowSubArray);   // from subArray to ProcessingUnit
			maxAddFromSubArray *= (int) ceil((double)(desiredTileSizeCM_x)/(double)(desiredPESizeCM_x));    // from ProcessingUnit to Tile
			if (param->parallelRead) {
				Gaccumulation->Initialize((int) maxTileAdded, ceil((double)log2((double)param->levelOutput))+param->numBitInput+1+ceil((double)log2((double)maxAddFromSubArray)),
										ceil((double)(desiredTileSizeCM_y)/(double)param->numColMuxed));
			} else {
				Gaccumulation->Initialize((int) maxTileAdded, ceil((double)log2((double)param->numRowSubArray)+(double)param->cellBit-1)+param->numBitInput+1+ceil((double)log2((double)maxAddFromSubArray)),
										ceil((double)(desiredTileSizeCM_y)/(double)param->numColMuxed));
			}
			if (param->reLu) {
				GreLu->Initialize(ceil((double)(desiredTileSizeCM_y)/(double)param->numColMuxed), param->numBitInput, param->clkFreq);
			} else {
				Gsigmoid->Initialize(false, param->numBitInput, ceil((double) log2((double) param->numRowSubArray)+(double) param->cellBit-1)+param->numBitInput+1+log2((double) maxAddFromSubArray)+ceil((double) log2((double) maxTileAdded)),
										ceil((double) (desiredTileSizeCM_y)/(double) param->numColMuxed), param->clkFreq);
			}
		}
	} else {   // activation inside tiles
		if (param->novelMapping) {
			int maxThroughputTile = (int) max((desiredTileSizeCM), ceil((double) sqrt((double) numPENM))*(double) (desiredPESizeNM));
			if (param->parallelRead) {
				Gaccumulation->Initialize((int) maxTileAdded, param->numBitInput, ceil((double) maxThroughputTile/(double) param->numColMuxed));
			} else {
				Gaccumulation->Initialize((int) maxTileAdded, param->numBitInput, ceil((double) maxThroughputTile/(double) param->numColMuxed));
			}
		}

		else {
			if (param->parallelRead) {
				Gaccumulation->Initialize((int) maxTileAdded, param->numBitInput, ceil((double) (desiredTileSizeCM_y)/(double) param->numColMuxed));
			} else {
				Gaccumulation->Initialize((int) maxTileAdded, param->numBitInput, ceil((double) (desiredTileSizeCM_y)/(double) param->numColMuxed));
			}
		}
	}
}



vector<double> ChipCalculateArea(InputParameter& inputParameter, Technology& tech, MemCell& cell, double desiredNumTileNM, double numPENM, double desiredPESizeNM, double desiredNumTileCM, double desiredTileSizeCM_x, double desiredTileSizeCM_y,
						double desiredPESizeCM_x, double desiredPESizeCM_y, int numTileRow, double *height, double *width, double *CMTileheight, double *CMTilewidth, double *NMTileheight, double *NMTilewidth) {

	vector<double> areaResults;

	double area = 0;
	double areaIC = 0;
	double area_tileIC = 0;
	double areaADC = 0;
	double areaAccum = 0;
	double areaOther = 0;
	double arealocalbuffer = 0;
	double areaglobalbuffer = 0;
	double areaotheralone = 0;
	double areasubarrayalone =0;


	double NMheight = 0;
	double NMwidth = 0;
	double CMheight = 0;
	double CMwidth = 0;

	*NMTileheight = 0;
	*NMTilewidth = 0;
	*CMTileheight = 0;
	*CMTilewidth = 0;
	*height = 0;
	*width = 0;

	vector<double> areaCMTile;
	vector<double> areaNMTile;

	if (param->novelMapping) {
		areaNMTile = TileCalculateArea(numPENM, desiredPESizeNM, &NMheight, &NMwidth);
		double NMTileArea = areaNMTile[0];
		double NMTileAreaIC = areaNMTile[1];
		double NMTileAreaADC = areaNMTile[2];
		double NMTileAreaAccum = areaNMTile[3];
		double NMTileAreaOther = areaNMTile[4];
		area += NMTileArea*desiredNumTileNM;
		areaIC += NMTileAreaIC*desiredNumTileNM;
		areaADC += NMTileAreaADC*desiredNumTileNM;
		areaAccum += NMTileAreaAccum*desiredNumTileNM;
		areaOther += NMTileAreaOther*desiredNumTileNM;

		*NMTileheight = NMheight;
		*NMTilewidth = NMwidth;
		//Addition Gokul Krishnan
		/*
		cout<<"The tile area in the chip is for Novel Mapping is "<<NMTileArea*1e12<< "um^2" << endl;
		cout<<"The routing area within the tile for the whole chip for Novel Mapping is "<<NMTileAreaIC*1e12<< "um^2" << endl;
		cout<<"The ADC area in the chip within the tile for Novel Mapping is "<<NMTileAreaADC*1e12<< "um^2" << endl;
		cout<<"The Accumulator area for the chip wihtin the tile for Novel Mapping is "<<NMTileAreaAccum*1e12<< "um^2" << endl;
		cout<<"The other periphery are in the tile for novel mapping is "<<NMTileAreaOther*1e12<< "um^2" << endl;
*/
	}

	areaCMTile = TileCalculateArea_new((((double)desiredTileSizeCM_x/(double) desiredPESizeCM_x)*((double)desiredTileSizeCM_y/(double) desiredPESizeCM_y))-1, desiredPESizeCM_x, desiredPESizeCM_y, &CMheight, &CMwidth);

	double CMTileArea = areaCMTile[0];
	double CMTileAreaIC = areaCMTile[1];
	double CMTileAreaADC = areaCMTile[2];
	double CMTileAreaAccum = areaCMTile[3];
	double CMTileAreaOther = areaCMTile[4];
	double CMTileAreaBuffer= areaCMTile[5];
	//double CMTileAreaotheralone= areaCMTile[6];
	double CMTilesubarrayareaalone=areaCMTile[6];
	/*
	cout<<"Single tile area in the chip is for conventional Mapping is "<<CMTileArea*1e12<< " um^2" << endl;
	cout<<"Single tile routing area within the tile for the whole chip for conventional Mapping is "<<CMTileAreaIC*1e12<< " um^2" << endl;
	cout<<"Single tile ADC area in the chip within the tile for conventional Mapping is "<<CMTileAreaADC*1e12<< " um^2" << endl;
	cout<<"Single tile Accumulator area for the chip wihtin the tile for conventional Mapping is "<<CMTileAreaAccum*1e12<< " um^2" << endl;
	cout<<"Single tile other periphery are in the tile for conventional mapping is "<<CMTileAreaOther*1e12<< " um^2" << endl;
	cout<<"Single tile buffer area of the tile is "<< CMTileAreaBuffer*1e12<<" um^2"<<endl;

	cout<<"The number of tiles is: "<<desiredNumTileCM<<endl;
*/
	area = CMTileArea*desiredNumTileCM;
	areaIC = CMTileAreaIC*desiredNumTileCM;

	areaADC = CMTileAreaADC*desiredNumTileCM;
	areaAccum = CMTileAreaAccum*desiredNumTileCM;
	areaOther = CMTileAreaOther*desiredNumTileCM;
	arealocalbuffer= CMTileAreaBuffer*desiredNumTileCM;
	//areaotheralone+= CMTileAreaotheralone*desiredNumTileCM;
	areasubarrayalone=CMTilesubarrayareaalone*desiredNumTileCM;


	*CMTileheight = CMheight;
	*CMTilewidth = CMwidth;
/*
	//Addition Gokul Krishnan
	cout<<"The tile area in the chip is for conventional Mapping is "<<area*1e12<< " um^2" << endl;
	cout<<"The routing area within the tile for the whole chip for conventional Mapping is "<<CMTileAreaIC*desiredNumTileCM*1e12<< " um^2" << endl;
	cout<<"The ADC area in the chip within the tile for conventional Mapping is "<<CMTileAreaADC*desiredNumTileCM*1e12<< " um^2" << endl;
	cout<<"The Accumulator area for the chip wihtin the tile for conventional Mapping is "<<CMTileAreaAccum*desiredNumTileCM*1e12<< " um^2" << endl;
	cout<<"The other periphery are in the tile for conventional mapping is "<<CMTileAreaOther*desiredNumTileCM*1e12<< " um^2" << endl;
	cout<<"The buffer area of the tile is "<< arealocalbuffer*1e12<<" um^2"<<endl;
*/
	globalBuffer->CalculateArea(numTileRow*max(NMheight, CMheight), NULL, NONE);
	GhTree->CalculateArea(max(NMheight, CMheight), max(NMwidth, CMwidth), param->treeFoldedRatio);
	maxPool->CalculateUnitArea(NONE);
	maxPool->CalculateArea(globalBuffer->width);
	Gaccumulation->CalculateArea(NULL, globalBuffer->height/3, NONE);

	areaglobalbuffer= globalBuffer->area;

	if (param->chipActivation) {
		if (param->reLu) {
			GreLu->CalculateArea(NULL, globalBuffer->width/3, NONE);
			area += GreLu->area;
		} else {
			Gsigmoid->CalculateUnitArea(NONE);
			Gsigmoid->CalculateArea(NULL, globalBuffer->width/3, NONE);
			area += Gsigmoid->area;
		}
	}

	//Addition Gokul Krishnan
	//cout<<"The area for the global routing of the global Htree is "<<GhTree->area*1e12<<endl;
	//cout<<"The area for the global buffer is "<<globalBuffer->area*1e12<<endl;
	//cout<<"The area for the maxpool is "<<maxPool->area*1e12<<endl;
	//cout<<"The area for the global accumulation is "<<Gaccumulation->area*1e12<<endl;

	area +=  maxPool->area + Gaccumulation->area; //+ GhTree->area;
	area_tileIC= areaIC;
	//areaIC += GhTree->area;

	areaResults.push_back(area);
	areaResults.push_back(areaIC);
	areaResults.push_back(areaADC);
	areaResults.push_back(areaAccum + Gaccumulation->area);
	areaResults.push_back(areaOther + maxPool->area);
	areaResults.push_back(arealocalbuffer);
	areaResults.push_back(areaglobalbuffer);
	//areaResults.push_back(areaOther + maxPool->area);
	areaResults.push_back(GhTree->area);
	areaResults.push_back(area_tileIC);
	areaResults.push_back(areasubarrayalone);

	*height = sqrt(area);
	*width = area/(*height);

	return areaResults;
}


double ChipCalculatePerformance(MemCell& cell, int layerNumber, const string &newweightfile, const string &oldweightfile, const string &inputfile, bool followedByMaxPool,
							const vector<vector<double> > &netStructure, const vector<int> &markNM, const vector<vector<double> > &numTileEachLayer, const vector<vector<double> > &utilizationEachLayer,
							const vector<vector<double> > &speedUpEachLayer, const vector<vector<double> > &tileLocaEachLayer, double numPENM, double desiredPESizeNM, double desiredTileSizeCM_x, double desiredTileSizeCM_y,
							double desiredPESizeCM_x, double desiredPESizeCM_y, double CMTileheight, double CMTilewidth, double NMTileheight, double NMTilewidth,
							double *readLatency, double *readDynamicEnergy, double *leakage, double *bufferLatency, double *bufferDynamicEnergy, double *icLatency, double *icDynamicEnergy,
							double *coreLatencyADC, double *coreLatencyAccum, double *coreLatencyOther, double *coreLatencyOther_only, double *coreEnergyOther_only, double *coreEnergyADC, double *coreEnergyAccum, double *coreEnergyOther, double *global_routinglatency, double *test, double *local_bufferlatency, double *local_bufferenergy
						, double *local_routinglatency, double *local_routingenergy, double *global_routingenergy, double *global_bufferlatency, double *global_bufferenergy) {


	int numRowPerSynapse, numColPerSynapse;
	numRowPerSynapse = param->numRowPerSynapse;
	numColPerSynapse = param->numColPerSynapse;

	// only get performance of single layer
	int l = layerNumber;
	// get weight matrix file Size
	int weightMatrixRow = netStructure[l][2]*netStructure[l][3]*netStructure[l][4]*numRowPerSynapse;
	int weightMatrixCol = netStructure[l][5]*numColPerSynapse;

	// load in whole file
	vector<vector<double> > inputVector;
	inputVector = LoadInInputData(inputfile);
	vector<vector<double> > newMemory;
	newMemory = LoadInWeightData(newweightfile, numRowPerSynapse, numColPerSynapse, param->maxConductance, param->minConductance);

	*readLatency = 0;
	*readDynamicEnergy = 0;
	*leakage = 0;
	*bufferLatency = 0;
	*bufferDynamicEnergy = 0;
	*icLatency = 0;
	*icDynamicEnergy = 0;

	*local_routinglatency=0;
	*global_routinglatency= 0;
	*local_bufferlatency = 0;
	*global_bufferlatency =0;
	*local_bufferenergy=0;
	*global_routingenergy=0;
	*global_routingenergy =0;
	*global_bufferenergy =0;



	*coreEnergyADC = 0;
	*coreEnergyAccum = 0;
	*coreEnergyOther = 0;
	*coreLatencyADC = 0;
	*coreLatencyAccum = 0;
	//*Global_accum_est_energy=0;
	//*Global_accum_est_lat = 0;
	*coreLatencyOther = 0;
	*coreLatencyOther_only = 0;
	*coreEnergyOther_only = 0;
	*test=0;

	double tileLeakage = 0;



	if (markNM[l] == 0) {   // conventional mapping
		//cout<<"Entering the conventional mapping loops"<<endl;
		double tileReadLatency = 0;
		double tileReadDynamicEnergy = 0;
		double tilebufferLatency = 0;
		double tilebufferDynamicEnergy = 0;
		double tileicLatency = 0;
		double tileicDynamicEnergy = 0;
		double tileLatencyADC = 0;
		double tileLatencyAccum = 0;
		double tileLatencyOther = 0;
		double tileEnergyADC = 0;
		double tileEnergyAccum = 0;
		double tileEnergyOther = 0;
		//double test=0;
		for (int i=0; i<numTileEachLayer[0][l]; i++) {       // # of tiles in row
			for (int j=0; j<numTileEachLayer[1][l]; j++) {   // # of tiles in Column

				//cout<<" Row: "<<i << endl;
				//cout<<" Column: "<<j <<endl;

				int numRowMatrix = min(desiredTileSizeCM_x, weightMatrixRow-i*desiredTileSizeCM_x);
				int numColMatrix = min(desiredTileSizeCM_y, weightMatrixCol-j*desiredTileSizeCM_y);
				double num, num_1;
				num_1 = (netStructure[l][0]-netStructure[l][3]+1)*(netStructure[l][1]-netStructure[l][4]+1);
				num = MAX(1,num_1)*param->numBitInput;
				// assign weight and input to specific tile
				vector<vector<double> > tileMemory;
				tileMemory = CopyArray(newMemory, i*desiredTileSizeCM_x, j*desiredTileSizeCM_y, numRowMatrix, numColMatrix);

				vector<vector<double> > tileInput;

				if (num_1 ==0){
					tileInput = CopyInput(inputVector, i*desiredTileSizeCM_x, num, numRowMatrix);

				}
				else{
					tileInput = CopyInput(inputVector, i*desiredTileSizeCM_x, (netStructure[l][0]-netStructure[l][3]+1)*(netStructure[l][1]-netStructure[l][4]+1)*param->numBitInput, numRowMatrix);

				}


				//cout<<"\n num is:"<<(num)<<endl;
				TileCalculatePerformance_new(tileMemory, tileMemory, tileInput, markNM[l], ceil((double)desiredTileSizeCM_x/(double)desiredPESizeCM_x) + ceil((double)desiredTileSizeCM_y/(double)desiredPESizeCM_y)-2, desiredPESizeCM_x, desiredPESizeCM_y, speedUpEachLayer[0][l], speedUpEachLayer[1][l],
									numRowMatrix, numColMatrix, num, cell, &tileReadLatency, &tileReadDynamicEnergy, &tileLeakage,
									&tilebufferLatency, &tilebufferDynamicEnergy, &tileicLatency, &tileicDynamicEnergy,
									&tileLatencyADC, &tileLatencyAccum, &tileLatencyOther, &tileEnergyADC, &tileEnergyAccum, &tileEnergyOther);


				*test = max(tileicLatency, (*test));
				*readLatency = max(tileReadLatency, (*readLatency));
				*readDynamicEnergy += tileReadDynamicEnergy;
				*bufferLatency = max(tilebufferLatency, (*bufferLatency));
				*bufferDynamicEnergy += tilebufferDynamicEnergy;
				*icLatency = max(tileicLatency, (*icLatency));

				*icDynamicEnergy += tileicDynamicEnergy;
				//if (tileLatencyADC ==0){
					//cout<<"The tile ADC latency is 0"<<endl;
				//}
				*coreLatencyADC = MAX(tileLatencyADC, (*coreLatencyADC));

				//cout<<"\n The Tile latency accum is: "<<tileLatencyAccum<<endl;
				*coreLatencyAccum = MAX(tileLatencyAccum, (*coreLatencyAccum));
				*coreLatencyOther = MAX(tileLatencyOther, (*coreLatencyOther));
				*coreLatencyOther_only = MAX(tileLatencyOther, (*coreLatencyOther_only));


				*coreEnergyADC += tileEnergyADC;
				*coreEnergyAccum += tileEnergyAccum;
				*coreEnergyOther += tileEnergyOther;
				*coreEnergyOther_only +=tileEnergyOther;

				if (param->chipActivation) {
					if (param->reLu) {
						GreLu->CalculateLatency(ceil((double) numTileEachLayer[0][l+1]*(double) numTileEachLayer[1][l+1]/(double) GreLu->numUnit));
						GreLu->CalculatePower(ceil((double) numTileEachLayer[0][l+1]*(double) numTileEachLayer[1][l+1]/(double) GreLu->numUnit));
						*readLatency += GreLu->readLatency;
						*readDynamicEnergy += GreLu->readDynamicEnergy;
						*coreLatencyOther += GreLu->readLatency;
						*coreLatencyOther_only += GreLu->readLatency;
						*coreEnergyOther += GreLu->readDynamicEnergy;
						*coreEnergyOther_only += GreLu->readDynamicEnergy;
					} else {
						Gsigmoid->CalculateLatency(ceil(numTileEachLayer[0][l+1]*numTileEachLayer[1][l+1]/Gsigmoid->numEntry));
						Gsigmoid->CalculatePower(ceil(numTileEachLayer[0][l+1]*numTileEachLayer[1][l+1]/Gsigmoid->numEntry));
						*readLatency += Gsigmoid->readLatency;
						*readDynamicEnergy += Gsigmoid->readDynamicEnergy;
						*coreLatencyOther += Gsigmoid->readLatency;
						*coreLatencyOther_only +=  Gsigmoid->readLatency;
						*coreEnergyOther += Gsigmoid->readDynamicEnergy;
						*coreEnergyOther_only += Gsigmoid->readDynamicEnergy;
					}
				}


				//cout<<"\n The Global accumulation latency accum is: "<<Gaccumulation->readLatency<<endl;
				// if this layer is followed by Max Pool
				if (followedByMaxPool) {
					int pass_to_maxpool = max(desiredTileSizeCM_x, desiredTileSizeCM_y);
					maxPool->CalculateLatency(1e20, 0, ceil((double) pass_to_maxpool/(double) (netStructure[l+1][0]*netStructure[l+1][1]/(double) maxPool->window)));
					maxPool->CalculatePower(ceil((double) pass_to_maxpool/(double) (netStructure[l+1][0]*netStructure[l+1][1]/maxPool->window)));
					*readLatency += maxPool->readLatency;
					*readDynamicEnergy += maxPool->readDynamicEnergy;
					*coreLatencyOther += maxPool->readLatency;
					*coreLatencyOther_only += maxPool->readLatency;
					*coreEnergyOther += maxPool->readDynamicEnergy;
					*coreEnergyOther_only += maxPool->readDynamicEnergy;
				}

			}

		}

		if (numTileEachLayer[0][l] > 1) {
			Gaccumulation->CalculateLatency(numTileEachLayer[1][l]*param->numColMuxed*(numTileEachLayer[0][l+1]*numTileEachLayer[1][l+1]), numTileEachLayer[0][l], 0);
			Gaccumulation->CalculatePower(numTileEachLayer[1][l]*param->numColMuxed*(numTileEachLayer[0][l+1]*numTileEachLayer[1][l+1]), numTileEachLayer[0][l]);
			*readLatency += Gaccumulation->readLatency;
			*readDynamicEnergy += Gaccumulation->readDynamicEnergy;
			*coreLatencyAccum += Gaccumulation->readLatency;
			*coreEnergyAccum += Gaccumulation->readDynamicEnergy;
			//*coreLatencyAccum += Gaccumulation->readLatency;
			//*coreEnergyAccum += Gaccumulation->readDynamicEnergy;
		}


/*
		//Addition Gokul Krishnan
		cout<<"The routing latency whicch includes bus plus the H tree within the tile is "<<tileicLatency*1e9<< "ns" << endl;
		//cout<<"The *iclatency taken is : "<<(test)*1e9<<" ns"<<endl;
		cout<<"The buffer latency wihtin the tile for the whole chip is "<<tilebufferLatency*1e9<< "ns" << endl;
		cout<<"The read latency within tile for the whole chip is "<<tileReadLatency*1e9<< "ns" << endl;
		cout<<"The ADC latency whithin the tile is "<<tileLatencyADC*1e9<< "ns" << endl;
		cout<<"The accumulator latency within the tile is "<<tileLatencyAccum*1e9<< "ns" << endl;
		cout<<"The other core latency for the tile is "<<tileLatencyOther*1e9<< "ns" << endl;
		cout<<"The global accumulation latency is: "<<Gaccumulation->readLatency*1e9<< "ns" << endl;


		cout<<"The routing energy whicch includes bus plus the H tree within the tile is "<<tileicDynamicEnergy*1e12<< "pJ" << endl;
		cout<<"The buffer energy wihtin the tile for the whole chip is "<<tilebufferDynamicEnergy*1e12<< "pJ" << endl;
		cout<<"The read energy within tile for the whole chip is "<<tileReadDynamicEnergy*1e12<< "pJ" << endl;
		cout<<"The ADC energy whithin the tile is "<<tileEnergyADC*1e12<< "pJ" << endl;
		cout<<"The accumulator energy within the tile is "<<tileEnergyAccum*1e12<< "pJ" << endl;
		cout<<"The other core energy for the tile is "<<tileEnergyOther*1e12<< "pJ" << endl;
*/
		//TODO
		int test, test_1;
		test_1 = (netStructure[l][0]-netStructure[l][3]+1)*(netStructure[l][1]-netStructure[l][4]+1);
		test = MAX(test_1, 1);

		GhTree->CalculateLatency(0, 0, tileLocaEachLayer[0][l], tileLocaEachLayer[1][l], CMTileheight, CMTilewidth,
								(weightMatrixRow+weightMatrixCol)*(test)/GhTree->busWidth);
		GhTree->CalculatePower(0, 0, tileLocaEachLayer[0][l], tileLocaEachLayer[1][l], CMTileheight, CMTilewidth, GhTree->busWidth,
								((weightMatrixRow)/desiredPESizeCM_x+(weightMatrixCol)/(desiredPESizeCM_y))*(test)/GhTree->busWidth);

		globalBuffer->CalculateLatency(weightMatrixRow*param->numBitInput, (test),
								weightMatrixCol*param->numBitInput, (test));
		globalBuffer->CalculatePower(weightMatrixRow*param->numBitInput, (test),
								weightMatrixCol*param->numBitInput, (test));

		*local_bufferlatency = *bufferLatency;//******************************************************************************************
		*local_bufferenergy = *bufferDynamicEnergy;//******************************************************************************************
		//*bufferLatency += globalBuffer->readLatency + globalBuffer->writeLatency;
		//*bufferDynamicEnergy += globalBuffer->readDynamicEnergy + globalBuffer->writeDynamicEnergy;
		*local_routinglatency = *icLatency;//******************************************************************************************
		*local_routingenergy = *icDynamicEnergy;//******************************************************************************************
		//*icLatency += GhTree->readLatency;
		//*icDynamicEnergy += GhTree->readDynamicEnergy;

		//*readLatency += globalBuffer->readLatency + globalBuffer->writeLatency + GhTree->readLatency;
		//*readDynamicEnergy += globalBuffer->readDynamicEnergy + globalBuffer->writeDynamicEnergy + GhTree->readDynamicEnergy;
		//cout<<"\n coreLatencyOther at chip.cpp level is: "<<*coreLatencyOther<<"\n"<<endl;
		//*coreLatencyOther += globalBuffer->readLatency + globalBuffer->writeLatency + GhTree->readLatency;
		*coreEnergyOther_only = *coreEnergyOther;//******************************************************************************************
		//*coreEnergyOther += globalBuffer->readDynamicEnergy + globalBuffer->writeDynamicEnergy + GhTree->readDynamicEnergy;

		//Addition Gokul Krishnan
		//cout<<"The global Htree read latency for is "<<GhTree->readLatency*1e9<<"ns"<<endl;
		//cout<<"The global buffer latency for is "<<(globalBuffer->readLatency + globalBuffer->writeLatency)*1e9<<"ns"<<endl;
		//cout<<"The global Htree dynamic energy is "<<GhTree->readDynamicEnergy*1e12<<"pJ"<<endl;
		//cout<<"The global buffer dynamic energy is "<<(globalBuffer->readDynamicEnergy + globalBuffer->writeDynamicEnergy)*1e12<<"pJ"<<endl;


		*global_routinglatency = GhTree->readLatency;//******************************************************************************************
		*global_routingenergy = GhTree->readDynamicEnergy;//******************************************************************************************
		*global_bufferlatency = globalBuffer->readLatency + globalBuffer->writeLatency;//******************************************************************************************
		*global_bufferenergy  = globalBuffer->readDynamicEnergy + globalBuffer->writeDynamicEnergy;//******************************************************************************************


	} else {   // novel Mapping
		cout<<"Entering the novel mapping loops"<<endl;
		double tileReadLatency = 0;
		double tileReadDynamicEnergy = 0;
		double tilebufferLatency = 0;
		double tilebufferDynamicEnergy = 0;
		double tileicLatency = 0;
		double tileicDynamicEnergy = 0;
		double tileLatencyADC = 0;
		double tileLatencyAccum = 0;
		double tileLatencyOther = 0;
		double tileEnergyADC = 0;
		double tileEnergyAccum = 0;
		double tileEnergyOther = 0;
		for (int i=0; i<numTileEachLayer[0][l]; i++) {       // # of tiles in row
			for (int j=0; j<numTileEachLayer[1][l]; j++) {   // # of tiles in Column



				int numRowMatrix = netStructure[l][2]*netStructure[l][3]*netStructure[l][4]*numRowPerSynapse/numTileEachLayer[0][l];
				int numColMatrix = netStructure[l][5]*numRowPerSynapse/numTileEachLayer[1][l];

				// assign weight and input to specific tile
				vector<vector<double> > tileMemory;
				tileMemory = ReshapeArray(newMemory, i*desiredPESizeNM, j*desiredPESizeNM, (int) netStructure[l][2]*numRowPerSynapse/numTileEachLayer[0][l],
									(int) netStructure[l][5]*numRowPerSynapse/numTileEachLayer[1][l], numPENM, (int) netStructure[l][2]*numRowPerSynapse);

				vector<vector<double> > tileInput;
				tileInput = ReshapeInput(inputVector, i*desiredPESizeNM, (int) (netStructure[l][0]-netStructure[l][3]+1)*(netStructure[l][1]-netStructure[l][4]+1)*param->numBitInput,
									(int) netStructure[l][2]*numRowPerSynapse/numTileEachLayer[0][l], numPENM, (int) netStructure[l][2]*numRowPerSynapse);

				TileCalculatePerformance(tileMemory, tileMemory, tileInput, markNM[l], numPENM, desiredPESizeNM, speedUpEachLayer[0][l], speedUpEachLayer[1][l],
									numRowMatrix, numColMatrix, (netStructure[l][0]-netStructure[l][3]+1)*(netStructure[l][1]-netStructure[l][4]+1)*param->numBitInput, cell,
									&tileReadLatency, &tileReadDynamicEnergy, &tileLeakage, &tilebufferLatency, &tilebufferDynamicEnergy, &tileicLatency, &tileicDynamicEnergy,
									&tileLatencyADC, &tileLatencyAccum, &tileLatencyOther, &tileEnergyADC, &tileEnergyAccum, &tileEnergyOther);


				*test = max(tileicLatency, (*test));
				*readLatency = max(tileReadLatency, (*readLatency));
				*readDynamicEnergy += tileReadDynamicEnergy;
				*bufferLatency = max(tilebufferLatency, (*bufferLatency));
				*bufferDynamicEnergy += tilebufferDynamicEnergy;
				*icLatency = max(tileicLatency, (*icLatency));
				*icDynamicEnergy += tileicDynamicEnergy;

				*coreLatencyADC = MAX(tileLatencyADC, (*coreLatencyADC));
				*coreLatencyAccum = MAX(tileLatencyAccum, (*coreLatencyAccum));
				*coreLatencyOther = MAX(tileLatencyOther, (*coreLatencyOther));

				*coreEnergyADC += tileEnergyADC;
				*coreEnergyAccum += tileEnergyAccum;
				*coreEnergyOther += tileEnergyOther;

				if (param->chipActivation) {
					if (param->reLu) {
						GreLu->CalculateLatency(ceil((double) numTileEachLayer[0][l+1]*(double) numTileEachLayer[1][l+1]/(double) GreLu->numUnit));
						GreLu->CalculatePower(ceil((double) numTileEachLayer[0][l+1]*(double) numTileEachLayer[1][l+1]/(double) GreLu->numUnit));
						*readLatency += GreLu->readLatency;
						*readDynamicEnergy += GreLu->readDynamicEnergy;
						*coreLatencyOther += GreLu->readLatency;
						*coreEnergyOther += GreLu->readDynamicEnergy;
					} else {
						Gsigmoid->CalculateLatency(ceil(numTileEachLayer[0][l+1]*numTileEachLayer[1][l+1]/Gsigmoid->numEntry));
						Gsigmoid->CalculatePower(ceil(numTileEachLayer[0][l+1]*numTileEachLayer[1][l+1]/Gsigmoid->numEntry));
						*readLatency += Gsigmoid->readLatency;
						*readDynamicEnergy += Gsigmoid->readDynamicEnergy;
						*coreLatencyOther += Gsigmoid->readLatency;
						*coreEnergyOther += Gsigmoid->readDynamicEnergy;
					}
				}

				if (numTileEachLayer[0][l] > 1) {
					Gaccumulation->CalculateLatency(numTileEachLayer[1][l]*param->numColMuxed*(numTileEachLayer[0][l+1]*numTileEachLayer[1][l+1]), numTileEachLayer[0][l], 0);
					Gaccumulation->CalculatePower(numTileEachLayer[1][l]*param->numColMuxed*(numTileEachLayer[0][l+1]*numTileEachLayer[1][l+1]), numTileEachLayer[0][l]);
					*readLatency += Gaccumulation->readLatency;
					*readDynamicEnergy += Gaccumulation->readDynamicEnergy;
					*coreLatencyAccum += Gaccumulation->readLatency;
					*coreEnergyAccum += Gaccumulation->readDynamicEnergy;
				}

				// if this layer is followed by Max Pool
				if (followedByMaxPool) {
					maxPool->CalculateLatency(1e20, 0, ceil((double) desiredPESizeNM*sqrt((double) numPENM)/(double) (netStructure[l+1][0]*netStructure[l+1][1]/(double) maxPool->window)));
					maxPool->CalculatePower(ceil((double) desiredPESizeNM*sqrt((double) numPENM)/(double) (netStructure[l+1][0]*netStructure[l+1][1]/maxPool->window)));
					*readLatency += maxPool->readLatency;
					*readDynamicEnergy += maxPool->readDynamicEnergy;
					*coreLatencyOther += maxPool->readLatency;
					*coreEnergyOther += maxPool->readDynamicEnergy;
				}
			}

		}

		//Addition Gokul Krishnan
		cout<<"The routing latency whicch includes bus plus the H tree within the tile is "<<tileicLatency*1e9<< "ns" << endl;
		cout<<"The buffer latency wihtin the tile for the whole chip is "<<tilebufferLatency*1e9<< "ns" << endl;
		cout<<"The read latency within tile for the whole chip is "<<tileReadLatency*1e9<< "ns" << endl;
		cout<<"The ADC latency whithin the tile is "<<tileLatencyADC*1e9<< "ns" << endl;
		cout<<"The accumulator latency within the tile is "<<tileLatencyAccum*1e9<< "ns" << endl;
		cout<<"The other core latency for the tile is "<<tileLatencyOther*1e9<< "ns" << endl;


		cout<<"The routing energy whicch includes bus plus the H tree within the tile is "<<tileicDynamicEnergy*1e12<< "pJ" << endl;
		cout<<"The buffer energy wihtin the tile for the whole chip is "<<tilebufferDynamicEnergy*1e12<< "pJ" << endl;
		cout<<"The read energy within tile for the whole chip is "<<tileReadDynamicEnergy*1e12<< "pJ" << endl;
		cout<<"The ADC energy whithin the tile is "<<tileEnergyADC*1e12<< "pJ" << endl;
		cout<<"The accumulator energy within the tile is "<<tileEnergyAccum*1e12<< "pJ" << endl;
		cout<<"The other core energy for the tile is "<<tileEnergyOther*1e12<< "pJ" << endl;


		*coreLatencyOther -= (*bufferLatency);
		*coreLatencyOther -= (*icLatency);
		*readLatency -= ((*bufferLatency) + (*icLatency));

		GhTree->CalculateLatency(0, 0, tileLocaEachLayer[0][l], tileLocaEachLayer[1][l], NMTileheight, NMTilewidth,
								(weightMatrixRow+weightMatrixCol)*(netStructure[l][0]-netStructure[l][3]+1)*(netStructure[l][1]-netStructure[l][4]+1)/GhTree->busWidth/netStructure[l][3]);
		GhTree->CalculatePower(0, 0, tileLocaEachLayer[0][l], tileLocaEachLayer[1][l], NMTileheight, NMTilewidth, GhTree->busWidth,
								((weightMatrixRow)/desiredPESizeCM_x+(weightMatrixCol)/(desiredPESizeCM_y))*(netStructure[l][0]-netStructure[l][3]+1)*(netStructure[l][1]-netStructure[l][4]+1)/GhTree->busWidth/netStructure[l][3]);

		globalBuffer->CalculateLatency(weightMatrixRow*param->numBitInput, (netStructure[l][0]-netStructure[l][3]+1)*(netStructure[l][1]-netStructure[l][4]+1)/netStructure[l][3],
								weightMatrixCol*param->numBitInput, (netStructure[l][0]-netStructure[l][3]+1)*(netStructure[l][1]-netStructure[l][4]+1)/netStructure[l][3]);
		globalBuffer->CalculatePower(weightMatrixRow*param->numBitInput, (netStructure[l][0]-netStructure[l][3]+1)*(netStructure[l][1]-netStructure[l][4]+1)/netStructure[l][3],
								weightMatrixCol*param->numBitInput, (netStructure[l][0]-netStructure[l][3]+1)*(netStructure[l][1]-netStructure[l][4]+1)/netStructure[l][3]);

		//*bufferLatency += globalBuffer->readLatency + globalBuffer->writeLatency;
		//*bufferDynamicEnergy += globalBuffer->readDynamicEnergy + globalBuffer->writeDynamicEnergy;
		//*icLatency += GhTree->readLatency;
		//*icDynamicEnergy += GhTree->readDynamicEnergy;

		*bufferLatency /= netStructure[l][3];
		*icLatency /= netStructure[l][3];

		*readLatency += (*bufferLatency) + (*icLatency);
		//*readDynamicEnergy += globalBuffer->readDynamicEnergy + globalBuffer->writeDynamicEnergy + GhTree->readDynamicEnergy;

		*coreLatencyOther += (*bufferLatency) + (*icLatency);
		//*coreEnergyOther += globalBuffer->readDynamicEnergy + globalBuffer->writeDynamicEnergy + GhTree->readDynamicEnergy;

		//Addition Gokul Krishnan
		//cout<<"The global Htree read latency is "<<GhTree->readLatency*1e9<< "ns" << endl;
		//cout<<"The global buffer latency foris "<<(globalBuffer->readLatency + globalBuffer->writeLatency)*1e9<< "ns" << endl;
		//cout<<"The global Htree dynamic energy is "<<GhTree->readDynamicEnergy*1e12<< "pJ" << endl;
		//cout<<"The global buffer dynamic energy is "<<(globalBuffer->readDynamicEnergy + globalBuffer->writeDynamicEnergy)*1e12<< "pJ" << endl;

		*global_bufferlatency = GhTree->readLatency/netStructure[l][3];

	}
	*leakage = tileLeakage;
}



vector<double> TileDesignCM(double tileSize, const vector<int > &markNM, const vector<vector<double> > &netStructure, int numRowPerSynapse, int numColPerSynapse) {
	double numTileTotal = 0;
	double matrixTotalCM = 0;
	double utilization = 0;
	cout<<"The tile size taken for calculating the number of tile is : "<<tileSize<<endl;
	for (int i=0; i<netStructure.size(); i++) {
		cout<<"(double) netStructure[i][2] : "<<(double) netStructure[i][2]<<endl;
		cout<<"(double) netStructure[i][3] : "<<(double) netStructure[i][3]<<endl;
		cout<<"(double) netStructure[i][4] : "<<(double) netStructure[i][4]<<endl;
		cout<<"(double) netStructure[i][5] : "<<(double) netStructure[i][5]<<endl;

	}

	cout<<"numRowPerSynapse is : "<<numRowPerSynapse<<endl;
	cout<<"numColPerSynapse is : "<<numColPerSynapse<<endl;


	for (int i=0; i<netStructure.size(); i++) {
		if (markNM[i] == 0) {
			numTileTotal += ceil((double) netStructure[i][2]*(double) netStructure[i][3]*(double) netStructure[i][4]*(double) numRowPerSynapse/(double) tileSize) * ceil(netStructure[i][5]*numColPerSynapse/tileSize);
			matrixTotalCM += netStructure[i][2]*netStructure[i][3]*netStructure[i][4]*numRowPerSynapse*netStructure[i][5]*numColPerSynapse;
		}
	}
	//numTileTotal = 80;
	utilization = matrixTotalCM/(numTileTotal*tileSize*tileSize);
	vector<double> tileDesignCM;
	tileDesignCM.push_back(numTileTotal);
	// tileDesignCM[0] = numTileTotal; tileDesignCM[1] = matrixTotalCM; tileDesignCM = utilization
	tileDesignCM.push_back(matrixTotalCM);
	tileDesignCM.push_back(utilization);
	return tileDesignCM;
	tileDesignCM.clear();
}

//**************************************************************************************************************************

vector<double> TileDesignCM_new(double tileSize_x, double tileSize_y, const vector<int > &markNM, const vector<vector<double> > &netStructure, int numRowPerSynapse, int numColPerSynapse) {
	double numTileTotal = 0;
	double matrixTotalCM = 0;
	double utilization = 0;
	//cout<<"The tile size taken along rows for calculating the number of tile is : "<<tileSize_x<<endl;
	//cout<<"The tile size taken along col for calculating the number of tile is : "<<tileSize_y<<endl;

	//cout<<"numRowPerSynapse is : "<<numRowPerSynapse<<endl;
	//cout<<"numColPerSynapse is : "<<numColPerSynapse<<endl;


	for (int i=0; i<netStructure.size(); i++) {
		//if (markNM[i] == 0) {
			if (netStructure[i][0]<netStructure[i][3]){
			numTileTotal += ceil((double) netStructure[i][2]*(double) netStructure[i][0]*(double) netStructure[i][1]*(double) numRowPerSynapse/(double) tileSize_x) * ceil(netStructure[i][5]*numColPerSynapse/(double) tileSize_y);
			matrixTotalCM += netStructure[i][2]*netStructure[i][3]*netStructure[i][0]*numRowPerSynapse*netStructure[i][1]*numColPerSynapse;}
			else{
			numTileTotal += ceil((double) netStructure[i][2]*(double) netStructure[i][3]*(double) netStructure[i][4]*(double) numRowPerSynapse/(double) tileSize_x) * ceil(netStructure[i][5]*numColPerSynapse/(double) tileSize_y);
			matrixTotalCM += netStructure[i][2]*netStructure[i][3]*netStructure[i][4]*numRowPerSynapse*netStructure[i][5]*numColPerSynapse;}
		//}
	}
	//numTileTotal = 80;
	utilization = matrixTotalCM/(numTileTotal*tileSize_y*tileSize_x);
	vector<double> tileDesignCM;
	tileDesignCM.push_back(numTileTotal);
	// tileDesignCM[0] = numTileTotal; tileDesignCM[1] = matrixTotalCM; tileDesignCM = utilization
	tileDesignCM.push_back(matrixTotalCM);
	tileDesignCM.push_back(utilization);
	return tileDesignCM;
	tileDesignCM.clear();
}
//**************************************************************************************************************************

vector<double> TileDesignNM(double peSize, const vector<int > &markNM, const vector<vector<double> > &netStructure, int numRowPerSynapse, int numColPerSynapse, double numPENM){
	double numTileTotal = 0;
	double matrixTotalNM = 0;
	double utilization = 0;
	for (int i=0; i<netStructure.size(); i++) {
		if (markNM[i] == 1) {
			numTileTotal += ceil((double) netStructure[i][2]*(double) numRowPerSynapse/(double) peSize) * ceil((double) netStructure[i][5]*(double) numColPerSynapse/(double) peSize);
			matrixTotalNM += netStructure[i][2]*netStructure[i][3]*netStructure[i][4]*numRowPerSynapse*netStructure[i][5]*numColPerSynapse;
		}
	}
	utilization = matrixTotalNM/(numTileTotal*peSize*peSize*numPENM);
	vector<double> tileDesignNM;
	// tileDesignNM[0] = numTileTotal; tileDesignNM[1] = matrixTotalNM; tileDesignNM = utilization
	tileDesignNM.push_back(numTileTotal);
	tileDesignNM.push_back(matrixTotalNM);
	tileDesignNM.push_back(utilization);
	return tileDesignNM;
	tileDesignNM.clear();
}

vector<vector<double> > PEDesign(bool Design, double peSize, double desiredTileSize, double numTileTotal, const vector<int > &markNM, const vector<vector<double> > &netStructure, int numRowPerSynapse, int numColPerSynapse) {
	double matrixTotalCM = 0;
	double utilization = 0;
	vector<double> peDupRow;
	vector<double> peDupCol;
	for (int i=0; i<netStructure.size(); i++) {
		int actualDupRow = 0;
		int actualDupCol = 0;
		if (markNM[i] ==0) {
			if ( (netStructure[i][2]*netStructure[i][3]*netStructure[i][4]*numRowPerSynapse <= desiredTileSize)||(netStructure[i][5]*numColPerSynapse <= desiredTileSize) ) {
				int peForOneMatrixRow = ceil((double) netStructure[i][2]*(double) netStructure[i][3]*(double) netStructure[i][4]*(double) numRowPerSynapse/(double) peSize);
				int peForOneMatrixCol = ceil((double) netStructure[i][5]*(double) numColPerSynapse/(double) peSize);
				int numPERow = ceil((double) desiredTileSize/(double) peSize);
				int numPECol = ceil((double) desiredTileSize/(double) peSize);
				actualDupRow = ceil((double) numPERow/(double) peForOneMatrixRow);
				actualDupCol = ceil((double) numPECol/(double) peForOneMatrixCol);
				matrixTotalCM += actualDupRow*actualDupCol*netStructure[i][2]*netStructure[i][3]*netStructure[i][4]*numRowPerSynapse*netStructure[i][5]*numColPerSynapse;
			} else {
				actualDupRow = 1;
				actualDupCol = 1;
				matrixTotalCM += actualDupRow*actualDupCol*netStructure[i][2]*netStructure[i][3]*netStructure[i][4]*numRowPerSynapse*netStructure[i][5]*numColPerSynapse;
			}
		} else {
			actualDupRow = 1;
			actualDupCol = 1;
		}
		peDupRow.push_back(actualDupRow);
		peDupCol.push_back(actualDupCol);
	}
	utilization = matrixTotalCM/(numTileTotal*desiredTileSize*desiredTileSize);
	vector<double> matrixTotal;
	matrixTotal.push_back(matrixTotalCM);
	vector<double> utiliz;
	utiliz.push_back(utilization);
	vector<vector<double> > peDesignCM;
	peDesignCM.push_back(matrixTotal);
	peDesignCM.push_back(utiliz);
	matrixTotal.clear();
	utiliz.clear();

	vector<vector<double> > peDup;
	peDup.push_back(peDupRow);
	peDup.push_back(peDupCol);
	peDupRow.clear();
	peDupCol.clear();
	if (Design) {
		return peDesignCM;
	} else {
		return peDup;
	}
	peDesignCM.clear();
	peDup.clear();
}

//*****************************************************************************************************************
vector<vector<double> > PEDesign_new(bool Design, double peSize_x, double peSize_y, double desiredTileSize_x, double desiredTileSize_y, double numTileTotal, const vector<int > &markNM, const vector<vector<double> > &netStructure, int numRowPerSynapse, int numColPerSynapse) {
	double matrixTotalCM = 0;
	double utilization = 0;
	vector<double> peDupRow;
	vector<double> peDupCol;
	for (int i=0; i<netStructure.size(); i++) {
		int actualDupRow = 0;
		int actualDupCol = 0;
		if (markNM[i] ==0) {
			if ( (netStructure[i][2]*netStructure[i][3]*netStructure[i][4]*numRowPerSynapse <= desiredTileSize_x)||(netStructure[i][5]*numColPerSynapse <= desiredTileSize_y) ) {
				int peForOneMatrixRow = ceil((double) netStructure[i][2]*(double) netStructure[i][3]*(double) netStructure[i][4]*(double) numRowPerSynapse/(double) peSize_x);
				int peForOneMatrixCol = ceil((double) netStructure[i][5]*(double) numColPerSynapse/(double) peSize_y);
				int numPERow = ceil((double) desiredTileSize_x/(double) peSize_x);
				int numPECol = ceil((double) desiredTileSize_y/(double) peSize_y);
				actualDupRow = ceil((double) numPERow/(double) peForOneMatrixRow);
				actualDupCol = ceil((double) numPECol/(double) peForOneMatrixCol);
				matrixTotalCM += actualDupRow*actualDupCol*netStructure[i][2]*netStructure[i][3]*netStructure[i][4]*numRowPerSynapse*netStructure[i][5]*numColPerSynapse;
			} else {
				actualDupRow = 1;
				actualDupCol = 1;
				matrixTotalCM += actualDupRow*actualDupCol*netStructure[i][2]*netStructure[i][3]*netStructure[i][4]*numRowPerSynapse*netStructure[i][5]*numColPerSynapse;
			}
		} else {
			actualDupRow = 1;
			actualDupCol = 1;
		}
		peDupRow.push_back(actualDupRow);
		peDupCol.push_back(actualDupCol);
	}
	utilization = matrixTotalCM/(numTileTotal*desiredTileSize_x*desiredTileSize_y);
	vector<double> matrixTotal;
	matrixTotal.push_back(matrixTotalCM);
	vector<double> utiliz;
	utiliz.push_back(utilization);
	vector<vector<double> > peDesignCM;
	peDesignCM.push_back(matrixTotal);
	peDesignCM.push_back(utiliz);
	matrixTotal.clear();
	utiliz.clear();

	vector<vector<double> > peDup;
	peDup.push_back(peDupRow);
	peDup.push_back(peDupCol);
	peDupRow.clear();
	peDupCol.clear();
	if (Design) {
		return peDesignCM;
	} else {
		return peDup;
	}
	peDesignCM.clear();
	peDup.clear();
}
//*****************************************************************************************************************

vector<vector<double> > SubArrayDup(double desiredPESizeCM, double desiredPESizeNM, const vector<int > &markNM, const vector<vector<double> > &netStructure, int numRowPerSynapse, int numColPerSynapse) {
	vector<double> subArrayDupRow;
	vector<double> subArrayDupCol;

	for (int i=0; i<netStructure.size(); i++) {
		int actualDupRow = 0;
		int actualDupCol = 0;
		if (markNM[i] == 0){
			if ( (netStructure[i][2]*netStructure[i][3]*netStructure[i][4]*numRowPerSynapse <= desiredPESizeCM)||(netStructure[i][5]*numColPerSynapse <= desiredPESizeCM) ) {
				int arrayForOneMatrixRow = ceil((double) netStructure[i][2]*(double) netStructure[i][3]*(double) netStructure[i][4]*(double) numRowPerSynapse/(double) param->numRowSubArray);
				int arrayForOneMatrixCol = ceil((double) netStructure[i][5]*(double) numColPerSynapse/(double) param->numColSubArray);
				int numSubArrayRow = ceil((double) desiredPESizeCM/(double) param->numRowSubArray);
				int numSubArrayCol = ceil((double) desiredPESizeCM/(double) param->numColSubArray);
				actualDupRow = ceil((double) numSubArrayRow/(double) arrayForOneMatrixRow);
				actualDupCol = ceil((double) numSubArrayCol/(double) arrayForOneMatrixCol);
			} else {
				actualDupRow = 1;
				actualDupCol = 1;
			}
		} else {
			if ( (netStructure[i][2]*numRowPerSynapse <= desiredPESizeNM)||(netStructure[i][5]*numColPerSynapse <= desiredPESizeNM) ) {
				int arrayForOneMatrixRow = ceil((double) netStructure[i][2]*(double) numRowPerSynapse/(double) param->numRowSubArray);
				int arrayForOneMatrixCol = ceil((double) netStructure[i][5]*(double) numColPerSynapse/(double) param->numColSubArray);
				int numSubArrayRow = ceil((double) desiredPESizeCM/(double) param->numRowSubArray);
				int numSubArrayCol = ceil((double) desiredPESizeCM/(double) param->numColSubArray);
				actualDupRow = ceil((double) numSubArrayRow/(double) arrayForOneMatrixRow);
				actualDupCol = ceil((double) numSubArrayCol/(double) arrayForOneMatrixCol);
			} else {
				actualDupRow = 1;
				actualDupCol = 1;
			}
		}
		subArrayDupRow.push_back(actualDupRow);
		subArrayDupCol.push_back(actualDupCol);
	}
	vector<vector<double> > subArrayDup;
	subArrayDup.push_back(subArrayDupRow);
	subArrayDup.push_back(subArrayDupCol);
	subArrayDupRow.clear();
	subArrayDupCol.clear();
	return subArrayDup;
	subArrayDup.clear();
}

//*****************************************************************************************************************
vector<vector<double> > SubArrayDup_new(double desiredPESizeCM_x, double desiredPESizeCM_y, double desiredPESizeNM, const vector<int > &markNM, const vector<vector<double> > &netStructure, int numRowPerSynapse, int numColPerSynapse) {
	vector<double> subArrayDupRow;
	vector<double> subArrayDupCol;

	for (int i=0; i<netStructure.size(); i++) {
		int actualDupRow = 0;
		int actualDupCol = 0;
		if (markNM[i] == 0){
			if ( (netStructure[i][2]*netStructure[i][3]*netStructure[i][4]*numRowPerSynapse <= desiredPESizeCM_x)||(netStructure[i][5]*numColPerSynapse <= desiredPESizeCM_y) ) {
				int arrayForOneMatrixRow = ceil((double) netStructure[i][2]*(double) netStructure[i][3]*(double) netStructure[i][4]*(double) numRowPerSynapse/(double) param->numRowSubArray);
				int arrayForOneMatrixCol = ceil((double) netStructure[i][5]*(double) numColPerSynapse/(double) param->numColSubArray);
				int numSubArrayRow = ceil((double) desiredPESizeCM_x/(double) param->numRowSubArray);
				int numSubArrayCol = ceil((double) desiredPESizeCM_y/(double) param->numColSubArray);
				actualDupRow = ceil((double) numSubArrayRow/(double) arrayForOneMatrixRow);
				actualDupCol = ceil((double) numSubArrayCol/(double) arrayForOneMatrixCol);
			} else {
				actualDupRow = 1;
				actualDupCol = 1;
			}
		} else {
			if ( (netStructure[i][2]*numRowPerSynapse <= desiredPESizeNM)||(netStructure[i][5]*numColPerSynapse <= desiredPESizeNM) ) {
				int arrayForOneMatrixRow = ceil((double) netStructure[i][2]*(double) numRowPerSynapse/(double) param->numRowSubArray);
				int arrayForOneMatrixCol = ceil((double) netStructure[i][5]*(double) numColPerSynapse/(double) param->numColSubArray);
				int numSubArrayRow = ceil((double) desiredPESizeCM_x/(double) param->numRowSubArray);
				int numSubArrayCol = ceil((double) desiredPESizeCM_y/(double) param->numColSubArray);
				actualDupRow = ceil((double) numSubArrayRow/(double) arrayForOneMatrixRow);
				actualDupCol = ceil((double) numSubArrayCol/(double) arrayForOneMatrixCol);
			} else {
				actualDupRow = 1;
				actualDupCol = 1;
			}
		}
		subArrayDupRow.push_back(actualDupRow);
		subArrayDupCol.push_back(actualDupCol);
	}
	vector<vector<double> > subArrayDup;
	subArrayDup.push_back(subArrayDupRow);
	subArrayDup.push_back(subArrayDupCol);
	subArrayDupRow.clear();
	subArrayDupCol.clear();
	return subArrayDup;
	subArrayDup.clear();
}

//*****************************************************************************************************************


vector<vector<double> > OverallEachLayer(bool utilization, bool speedUp, const vector<vector<double> > &peDup, const vector<vector<double> > &subArrayDup, double desiredTileSizeCM, double desiredPESizeNM,
										const vector<int > &markNM, const vector<vector<double> > &netStructure, int numRowPerSynapse, int numColPerSynapse, double numPENM) {
	vector<double> numTileEachLayerRow;
	vector<double> numTileEachLayerCol;
	vector<vector<double> > utilizationEachLayer;
	vector<double> speedUpEachLayerRow;
	vector<double> speedUpEachLayerCol;

	for (int i=0; i<netStructure.size(); i++) {
		vector<double> utilization;

		double numtileEachLayerRow, numtileEachLayerCol, utilizationEach;
		if (markNM[i] == 0) {
			// conventional mapping
			numtileEachLayerRow = ceil((double) netStructure[i][2]*(double) netStructure[i][3]*(double) netStructure[i][4]*(double) numRowPerSynapse/desiredTileSizeCM);
			numtileEachLayerCol = ceil((double) netStructure[i][5]*(double) numColPerSynapse/(double) desiredTileSizeCM);
			utilizationEach = (peDup[0][i]*peDup[1][i]*subArrayDup[0][i]*subArrayDup[1][i]*netStructure[i][2]*netStructure[i][3]*netStructure[i][4]
										*numRowPerSynapse*netStructure[i][5]*numColPerSynapse)/(numtileEachLayerRow*numtileEachLayerCol*desiredTileSizeCM*desiredTileSizeCM);

			utilization.push_back(utilizationEach);
		} else {
			// novel mapping
			numtileEachLayerRow = ceil((double) netStructure[i][2]*(double) numRowPerSynapse/(double) desiredPESizeNM);
			numtileEachLayerCol = ceil((double) netStructure[i][5]*(double) numColPerSynapse/(double) desiredPESizeNM);
			utilizationEach = (peDup[0][i]*peDup[1][i]*subArrayDup[0][i]*subArrayDup[1][i]*netStructure[i][2]*numPENM*numRowPerSynapse*netStructure[i][5]
										*numColPerSynapse)/(numtileEachLayerRow*numtileEachLayerCol*desiredPESizeNM*desiredPESizeNM*numPENM);

			utilization.push_back(utilizationEach);
		}
		numTileEachLayerRow.push_back(numtileEachLayerRow);
		numTileEachLayerCol.push_back(numtileEachLayerCol);
		utilizationEachLayer.push_back(utilization);
		speedUpEachLayerRow.push_back(peDup[0][i]*subArrayDup[0][i]);
		speedUpEachLayerCol.push_back(peDup[1][i]*subArrayDup[1][i]);
		utilization.clear();
	}

	vector<vector<double> > numTileEachLayer;
	numTileEachLayer.push_back(numTileEachLayerRow);
	numTileEachLayer.push_back(numTileEachLayerCol);
	numTileEachLayerRow.clear();
	numTileEachLayerCol.clear();

	vector<vector<double> > speedUpEachLayer;
	speedUpEachLayer.push_back(speedUpEachLayerRow);
	speedUpEachLayer.push_back(speedUpEachLayerCol);
	speedUpEachLayerRow.clear();
	speedUpEachLayerCol.clear();

	if (utilization) {
		return utilizationEachLayer;
	} else if (speedUp) {
		return speedUpEachLayer;
	} else {
		return numTileEachLayer;
	}
	utilizationEachLayer.clear();
	speedUpEachLayer.clear();
	numTileEachLayer.clear();
}

//*************************************************************************************************************
vector<vector<double> > OverallEachLayer_new(bool utilization, bool speedUp, const vector<vector<double> > &peDup, const vector<vector<double> > &subArrayDup, double desiredTileSizeCM_x, double desiredTileSizeCM_y, double desiredPESizeNM,
										const vector<int > &markNM, const vector<vector<double> > &netStructure, int numRowPerSynapse, int numColPerSynapse, double numPENM) {
	vector<double> numTileEachLayerRow;
	vector<double> numTileEachLayerCol;
	vector<vector<double> > utilizationEachLayer;
	vector<double> speedUpEachLayerRow;
	vector<double> speedUpEachLayerCol;

	for (int i=0; i<netStructure.size(); i++) {
		vector<double> utilization;

		double numtileEachLayerRow, numtileEachLayerCol, utilizationEach;
		if (markNM[i] == 0) {
			// conventional mapping

				if (netStructure[i][0] < netStructure[i][3]) {

					numtileEachLayerRow = ceil((double) netStructure[i][2]*(double) (netStructure[i][0])*(double) netStructure[i][1]*(double) numRowPerSynapse/desiredTileSizeCM_x);
					numtileEachLayerCol = ceil((double) netStructure[i][5]*(double) numColPerSynapse/(double) desiredTileSizeCM_y);
					utilizationEach = (peDup[0][i]*peDup[1][i]*subArrayDup[0][i]*subArrayDup[1][i]*netStructure[i][2]*netStructure[i][0]*netStructure[i][1]
												*numRowPerSynapse*netStructure[i][5]*numColPerSynapse)/(numtileEachLayerRow*numtileEachLayerCol*desiredTileSizeCM_x*desiredTileSizeCM_y);

				} else {
					numtileEachLayerRow = ceil((double) netStructure[i][2]*(double) netStructure[i][3]*(double) netStructure[i][4]*(double) numRowPerSynapse/desiredTileSizeCM_x);
					numtileEachLayerCol = ceil((double) netStructure[i][5]*(double) numColPerSynapse/(double) desiredTileSizeCM_y);
					utilizationEach = (peDup[0][i]*peDup[1][i]*subArrayDup[0][i]*subArrayDup[1][i]*netStructure[i][2]*netStructure[i][3]*netStructure[i][4]
												*numRowPerSynapse*netStructure[i][5]*numColPerSynapse)/(numtileEachLayerRow*numtileEachLayerCol*desiredTileSizeCM_x*desiredTileSizeCM_y);
				}
			utilization.push_back(utilizationEach);
		} else {
			// novel mapping
			numtileEachLayerRow = ceil((double) netStructure[i][2]*(double) numRowPerSynapse/(double) desiredPESizeNM);
			numtileEachLayerCol = ceil((double) netStructure[i][5]*(double) numColPerSynapse/(double) desiredPESizeNM);
			utilizationEach = (peDup[0][i]*peDup[1][i]*subArrayDup[0][i]*subArrayDup[1][i]*netStructure[i][2]*numPENM*numRowPerSynapse*netStructure[i][5]
										*numColPerSynapse)/(numtileEachLayerRow*numtileEachLayerCol*desiredPESizeNM*desiredPESizeNM*numPENM);

			utilization.push_back(utilizationEach);
		}
		numTileEachLayerRow.push_back(numtileEachLayerRow);
		numTileEachLayerCol.push_back(numtileEachLayerCol);
		utilizationEachLayer.push_back(utilization);
		speedUpEachLayerRow.push_back(peDup[0][i]*subArrayDup[0][i]);
		speedUpEachLayerCol.push_back(peDup[1][i]*subArrayDup[1][i]);
		utilization.clear();
	}

	vector<vector<double> > numTileEachLayer;
	numTileEachLayer.push_back(numTileEachLayerRow);
	numTileEachLayer.push_back(numTileEachLayerCol);
	numTileEachLayerRow.clear();
	numTileEachLayerCol.clear();

	vector<vector<double> > speedUpEachLayer;
	speedUpEachLayer.push_back(speedUpEachLayerRow);
	speedUpEachLayer.push_back(speedUpEachLayerCol);
	speedUpEachLayerRow.clear();
	speedUpEachLayerCol.clear();

	if (utilization) {
		return utilizationEachLayer;
	} else if (speedUp) {
		return speedUpEachLayer;
	} else {
		return numTileEachLayer;
	}
	utilizationEachLayer.clear();
	speedUpEachLayer.clear();
	numTileEachLayer.clear();
}
//*************************************************************************************************************

vector<vector<double> > LoadInWeightData(const string &weightfile, int numRowPerSynapse, int numColPerSynapse, double maxConductance, double minConductance) {

	ifstream fileone(weightfile.c_str());
	string lineone;
	string valone;

	int ROW = 0;
	int COL = 0;

	if (!fileone.good()) {
		cerr << "Error: the fileone cannot be opened!" << endl;
		exit(1);
	}else{
		while (getline(fileone, lineone, '\n')) {
			ROW++;
		}
		fileone.clear();
		fileone.seekg(0, ios::beg);
		if (getline(fileone, lineone, '\n')) {
			istringstream iss (lineone);
			while (getline(iss, valone, ',')) {
				COL++;
			}
		}
	}
	fileone.clear();
	fileone.seekg(0, ios::beg);


	double NormalizedMin = 0;
	double NormalizedMax = pow(2, param->synapseBit)-1;

	double RealMax = param->algoWeightMax;
	double RealMin = param->algoWeightMin;

	vector<vector<double> > weight;
	// load the data into a weight matrix ...
	for (int row=0; row<ROW; row++) {
		vector<double> weightrow;
		vector<double> weightrowb;
		getline(fileone, lineone, '\n');
		istringstream iss;
		iss.str(lineone);
		for (int col=0; col<COL; col++) {
			while(getline(iss, valone, ',')){
				istringstream fs;
				fs.str(valone);
				double f=0;
				fs >> f;
				//normalize weight to integer
				double newdata = ((NormalizedMax-NormalizedMin)/(RealMax-RealMin)*(f-RealMax)+NormalizedMax);
				if (newdata >= 0) {
					newdata += 0.5;
				}else {
					newdata -= 0.5;
				}
				// map and expend the weight in memory array
				int cellrange = pow(2, param->cellBit);
				vector<double> synapsevector(numColPerSynapse);
				int value = newdata;
				if (param->BNNparallelMode) {
					if (value == 1) {
						weightrow.push_back(maxConductance);
						weightrow.push_back(minConductance);
					} else {
						weightrow.push_back(minConductance);
						weightrow.push_back(maxConductance);
					}
				} else if (param->XNORparallelMode || param->XNORsequentialMode) {
					if (value == 1) {
						weightrow.push_back(maxConductance);
						weightrowb.push_back(minConductance);
					} else {
						weightrow.push_back(minConductance);
						weightrowb.push_back(maxConductance);
					}
				} else {
					int remainder;
					for (int z=0; z<numColPerSynapse; z++) {
						remainder = ceil((double)(value%cellrange));
						value = ceil((double)(value/cellrange));
						synapsevector.insert(synapsevector.begin(), remainder);
					}
					for (int u=0; u<numColPerSynapse; u++) {
						double cellvalue = synapsevector[u];
						double conductance = cellvalue/(cellrange-1) * (maxConductance-minConductance) + minConductance;
						weightrow.push_back(conductance);
					}
				}
			}
		}
		if (param->XNORparallelMode || param->XNORsequentialMode) {
			weight.push_back(weightrow);
			weightrow.clear();
			weight.push_back(weightrowb);
			weightrowb.clear();
		} else {
			weight.push_back(weightrow);
			weightrow.clear();
		}
	}
	fileone.close();

	return weight;
	weight.clear();
}



vector<vector<double> > CopyArray(const vector<vector<double> > &orginal, int positionRow, int positionCol, int numRow, int numCol) {

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



vector<vector<double> > ReshapeArray(const vector<vector<double> > &orginal, int positionRow, int positionCol, int numRow, int numCol, int numPE, int weightMatrixRow) {

	vector<vector<double> > copy;

	for (int k=0; k<numPE; k++) {
		for (int i=0; i<numRow; i++) {
			vector<double> copyRow;
			for (int j=0; j<numCol; j++) {
				copyRow.push_back(orginal[positionRow+k*weightMatrixRow+i][positionCol+j]);
			}
			copy.push_back(copyRow);
			copyRow.clear();
		}
	}

	return copy;
	copy.clear();
}



vector<vector<double> > LoadInInputData(const string &inputfile) {

	ifstream infile(inputfile.c_str());
	string inputline;
	string inputval;


	int ROWin=0, COLin=0;
	if (!infile.good()) {
		cerr << "Error: the input file cannot be opened!" << endl;
		exit(1);
	}else{
		while (getline(infile, inputline, '\n')) {
			ROWin++;
		}
		infile.clear();
		infile.seekg(0, ios::beg);
		if (getline(infile, inputline, '\n')) {
			istringstream iss (inputline);
			while (getline(iss, inputval, ',')) {
				COLin++;
			}
		}
	}
	infile.clear();
	infile.seekg(0, ios::beg);

	vector<vector<double> > inputvector;
	// load the data into inputvector ...
	for (int row=0; row<ROWin; row++) {
		vector<double> inputvectorrow;
		vector<double> inputvectorrowb;
		getline(infile, inputline, '\n');
		istringstream iss;
		iss.str(inputline);
		for (int col=0; col<COLin; col++) {
			while(getline(iss, inputval, ',')){
				istringstream fs;
				fs.str(inputval);
				double f=0;
				fs >> f;

				if (param->BNNparallelMode) {
					if (f == 1) {
						inputvectorrow.push_back(1);
					} else {
						inputvectorrow.push_back(0);
					}
				} else if (param->XNORparallelMode || param->XNORsequentialMode) {
					if (f == 1) {
						inputvectorrow.push_back(1);
						inputvectorrowb.push_back(0);
					} else {
						inputvectorrow.push_back(0);
						inputvectorrowb.push_back(1);
					}
				} else {
					inputvectorrow.push_back(f);
				}
			}
		}
		if (param->XNORparallelMode || param->XNORsequentialMode) {
			inputvector.push_back(inputvectorrow);
			inputvectorrow.clear();
			inputvector.push_back(inputvectorrowb);
			inputvectorrowb.clear();
		} else {
			inputvector.push_back(inputvectorrow);
			inputvectorrow.clear();
		}
	}
	// close the input file ...
	infile.close();

	return inputvector;
	inputvector.clear();
}




vector<vector<double> > CopyInput(const vector<vector<double> > &orginal, int positionRow, int numInputVector, int numRow) {

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



vector<vector<double> > ReshapeInput(const vector<vector<double> > &orginal, int positionRow, int numInputVector, int numRow, int numPE, int weightMatrixRow) {

	vector<vector<double> > copy;

	for (int k=0; k<numPE; k++) {
		for (int i=0; i<numRow; i++) {
			vector<double> copyRow;
			for (int j=0; j<numInputVector; j++) {
				copyRow.push_back(orginal[positionRow+k*weightMatrixRow+i][j]);
			}
			copy.push_back(copyRow);
			copyRow.clear();
		}
	}

	return copy;
	copy.clear();
}
