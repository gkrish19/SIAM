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

#include <cstdio>
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <sstream>
#include <chrono>
#include <algorithm>
#include "constant.h"
#include "formula.h"
#include "Param.h"
#include "Tile.h"
#include "Chip.h"
#include "ProcessingUnit.h"
#include "SubArray.h"
#include "Definition.h"

using namespace std;

vector<vector<double> > getNetStructure(const string &inputfile);

int main(int argc, char * argv[]) {

	auto start = chrono::high_resolution_clock::now();

	gen.seed(0);

	vector<vector<double> > netStructure;
	netStructure = getNetStructure(argv[1]);

	// define weight/input/memory precision from wrapper
	param->synapseBit = atoi(argv[2]);              // precision of synapse weight
	param->numBitInput = atoi(argv[3]);             // precision of input neural activation
	if (param->cellBit > param->synapseBit) {
		cout << "ERROR!: Memory precision is even higher than synapse precision, please modify 'cellBit' in Param.cpp!" << endl;
		param->cellBit = param->synapseBit;
	}
	param->numColPerSynapse = ceil((double)param->synapseBit/(double)param->cellBit);
	param->numRowPerSynapse = 1;

	double maxPESizeNM, maxTileSizeCM_x, maxTileSizeCM_y, numPENM;
	vector<int> markNM;
	markNM = ChipDesignInitialize(inputParameter, tech, cell, netStructure, &maxPESizeNM, &maxTileSizeCM_x, &maxTileSizeCM_y, &numPENM);

	double desiredNumTileNM, desiredPESizeNM, desiredNumTileCM, desiredTileSizeCM, desiredTileSizeCM_x, desiredTileSizeCM_y, desiredPESizeCM, desiredPESizeCM_x, desiredPESizeCM_y;
	int numTileRow, numTileCol;

	vector<vector<double> > numTileEachLayer;
	vector<vector<double> > utilizationEachLayer;
	vector<vector<double> > speedUpEachLayer;
	vector<vector<double> > tileLocaEachLayer;

	numTileEachLayer = ChipFloorPlan(true, false, false, netStructure, markNM,
					maxPESizeNM, maxTileSizeCM_x, maxTileSizeCM_y, numPENM,
					&desiredNumTileNM, &desiredPESizeNM, &desiredNumTileCM, &desiredTileSizeCM_x, &desiredTileSizeCM_y, &desiredPESizeCM_x, &desiredPESizeCM_y, &numTileRow, &numTileCol);

	utilizationEachLayer = ChipFloorPlan(false, true, false, netStructure, markNM,
					maxPESizeNM, maxTileSizeCM_x, maxTileSizeCM_y, numPENM,
					&desiredNumTileNM, &desiredPESizeNM, &desiredNumTileCM, &desiredTileSizeCM_x, &desiredTileSizeCM_y, &desiredPESizeCM_x, &desiredPESizeCM_y, &numTileRow, &numTileCol);

	speedUpEachLayer = ChipFloorPlan(false, false, true, netStructure, markNM,
					maxPESizeNM, maxTileSizeCM_x, maxTileSizeCM_y, numPENM,
					&desiredNumTileNM, &desiredPESizeNM, &desiredNumTileCM, &desiredTileSizeCM_x, &desiredTileSizeCM_y, &desiredPESizeCM_x, &desiredPESizeCM_y, &numTileRow, &numTileCol);

	tileLocaEachLayer = ChipFloorPlan(false, false, false, netStructure, markNM,
					maxPESizeNM, maxTileSizeCM_x, maxTileSizeCM_y, numPENM,
					&desiredNumTileNM, &desiredPESizeNM, &desiredNumTileCM, &desiredTileSizeCM_x, &desiredTileSizeCM_y, &desiredPESizeCM_x, &desiredPESizeCM_y, &numTileRow, &numTileCol);

	cout << "------------------------------ FloorPlan --------------------------------" <<  endl;
	cout << endl;
	//cout << "Tile and PE size are optimized to maximize memory utilization ( = memory mapped by synapse / total memory on chip)" << endl;
	cout << endl;
	if (!param->novelMapping) {
		cout << "Desired Conventional Mapped Tile Storage Size: " << desiredTileSizeCM_x << "x" << desiredTileSizeCM_y << endl;
		cout << "Desired Conventional PE Storage Size: " << desiredPESizeCM_x << "x" << desiredPESizeCM_y << endl;
	} else {
		cout << "Desired Conventional Mapped Tile Storage Size: " << desiredTileSizeCM << "x" << desiredTileSizeCM << endl;
		cout << "Desired Conventional PE Storage Size: " << desiredPESizeCM << "x" << desiredPESizeCM << endl;
		cout << "Desired Novel Mapped Tile Storage Size: " << numPENM << "x" << desiredPESizeNM << "x" << desiredPESizeNM << endl;
	}
	cout << "User-defined SubArray Size: " << param->numRowSubArray << "x" << param->numColSubArray << endl;
	cout << endl;
	cout << "----------------- # of tile used for each layer -----------------" <<  endl;
	double totalNumTile = 0;

	std::ofstream myfile;
	myfile.open ("./to_interconnect/num_tiles_per_layer.csv"); //Dumps file for the number of tile per layer for the interconnect simulator.
	for (int i=0; i<netStructure.size(); i++) {
		cout << "layer" << i+1 << ": " << numTileEachLayer[0][i] * numTileEachLayer[1][i] << endl;
		//myfile << i+1 <<","<< numTileEachLayer[0][i] * numTileEachLayer[1][i] << endl;
		myfile << numTileEachLayer[0][i] * numTileEachLayer[1][i] << endl;
		totalNumTile += numTileEachLayer[0][i] * numTileEachLayer[1][i];
	}
	myfile.close();
	cout << endl;
/*
	cout << "----------------- Speed-up of each layer ------------------" <<  endl;
	for (int i=0; i<netStructure.size(); i++) {
		cout << "layer" << i+1 << ": " << speedUpEachLayer[0][i] << ", " << speedUpEachLayer[1][i] << endl;
	}*/
	cout << endl;

	cout << "----------------- Utilization of each layer ------------------" <<  endl;
	double realMappedMemory = 0;
	for (int i=0; i<netStructure.size(); i++) {
		cout << "layer" << i+1 << ": " << utilizationEachLayer[i][0] << endl;
		realMappedMemory += numTileEachLayer[0][i] * numTileEachLayer[1][i] * utilizationEachLayer[i][0];
	}
	cout<<"The total number of tiles are : "<<totalNumTile<<endl;
	cout<<"The real Mapped memory is : "<<realMappedMemory<<endl;
	cout << "Memory Utilization of Whole Chip: " << realMappedMemory/totalNumTile*100 << " % " << endl;
	cout << endl;
	cout << "---------------------------- FloorPlan Done ------------------------------" <<  endl;
	cout << endl;
	cout << endl;
	cout << endl;

	double numComputation = 0;
	for (int i=0; i<netStructure.size(); i++) {
		numComputation += netStructure[i][0] * netStructure[i][1] * netStructure[i][2] * netStructure[i][3] * netStructure[i][4] * netStructure[i][5];
	}

	//cout<<"The total number of computations are : "<<numComputation<<endl;

	ChipInitialize(inputParameter, tech, cell, netStructure, markNM, numTileEachLayer,
					numPENM, desiredNumTileNM, desiredPESizeNM, desiredNumTileCM, desiredTileSizeCM_x, desiredTileSizeCM_y, desiredPESizeCM_x, desiredPESizeCM_y, numTileRow, numTileCol);

	double chipHeight, chipWidth, chipArea, chipAreaIC, chipAreaADC, chipAreaAccum, chipAreaOther, chipArealocalbuffer,chipAreaglobalbuffer, chipAreaOtheralone, chipAreaglobalIC, chipAreatileIC, chipAreasubarrayalone;
	chipAreasubarrayalone=0;
	double CMTileheight = 0;
	double CMTilewidth = 0;
	double NMTileheight = 0;
	double NMTilewidth = 0;
	//double chipAreasubarrayalone =0;
	vector<double> chipAreaResults;

	chipAreaResults = ChipCalculateArea(inputParameter, tech, cell, desiredNumTileNM, numPENM, desiredPESizeNM, desiredNumTileCM, desiredTileSizeCM_x, desiredTileSizeCM_y, desiredPESizeCM_x, desiredPESizeCM_y, numTileRow,
					&chipHeight, &chipWidth, &CMTileheight, &CMTilewidth, &NMTileheight, &NMTilewidth);
	chipArea = chipAreaResults[0];
	chipAreaIC = chipAreaResults[1];
	chipAreaADC = chipAreaResults[2];
	chipAreaAccum = chipAreaResults[3];
	chipAreaOther = chipAreaResults[4];
	chipArealocalbuffer = chipAreaResults[5];
	chipAreaglobalbuffer = chipAreaResults[6];
	//chipAreaOtheralone = chipAreaResults[7];
	chipAreaglobalIC = chipAreaResults[7];
	chipAreatileIC = chipAreaResults[8];
	chipAreasubarrayalone=chipAreaResults[9];
	std::ofstream myarea;
	myarea.open ("./Final_Results/area.csv");
	myarea<<"Subarray Area"<<","<<chipAreasubarrayalone * 1e12<<","<<"um^2"<<endl;
	//cout<<"The total subarray (compute engine) only area is "<< chipAreasubarrayalone * 1e12 << " um^2"<<endl;

	double chipReadLatency = 0;
	double chipReadDynamicEnergy = 0;
	double chipLeakageEnergy = 0;
	double chipLeakage = 0;
	double chipbufferLatency = 0;
	double chipbufferReadDynamicEnergy = 0;
	//double chipAreasubarrayalone =0;
	double chipicLatency = 0;
	double chipicReadDynamicEnergy = 0;
	double global_iclatency = 0;

	double chipLatencyADC = 0;
	double chipLatencyAccum = 0;
	double chipLatencyOther = 0;
	double chipEnergyADC = 0;
	double chipEnergyAccum = 0;
	double chipEnergyOther = 0;

	double layerReadLatency = 0;
	/******************************************/
	//Added by Gokul Krishnan
	double avg_layerReadLatency = 0;
	int count = 0;
	/******************************************/
	double layerReadDynamicEnergy = 0;
	double tileLeakage = 0;
	double layerbufferLatency = 0;
	double layerbufferDynamicEnergy = 0;
	double layericLatency = 0;
	double layericDynamicEnergy = 0;
	double tile_total =0;
	double coreLatencyOther_only= 0;
	double coreEnergyOther_only = 0;
	double coreLatencyADC = 0;
	double coreLatencyAccum = 0;
	double coreLatencyOther = 0;
	double coreEnergyADC = 0;
	double coreEnergyAccum = 0;
	double coreEnergyOther = 0;

	double Global_accum_est_lat=0;
	double Global_accum_est_energy=0;
	double max_glob_acc_lat = 0;
	double max_glob_acc_energy = 0;

	double global_routinglatency_1= 0;
	double local_routinglatency_1= 0;
	double global_bufferlatency_1= 0;
	double local_bufferlatency_1= 0;
	double coreLatencyOther_only_1 =0;

	double global_bufferenergy_1= 0;
	double coreEnergyOther_only_1 =0;
	double local_bufferenergy_1 = 0;
	double global_routingenergy_1 = 0;
	double local_routingenergy_1 = 0;

	double global_routinglatency, test, local_bufferlatency, local_bufferenergy, local_routinglatency, local_routingenergy, global_routingenergy, global_bufferlatency, global_bufferenergy = 0;

	cout << "-------------------------------------- Hardware Performance --------------------------------------" <<  endl;

	for (int i=0; i<netStructure.size(); i++) {

		cout << "-------------------- Estimation of Layer " << i+1 << " ----------------------" << endl;

		ChipCalculatePerformance(cell, i, argv[2*i+4], argv[2*i+4], argv[2*i+5], netStructure[i][6],
					netStructure, markNM, numTileEachLayer, utilizationEachLayer, speedUpEachLayer, tileLocaEachLayer,
					numPENM, desiredPESizeNM, desiredTileSizeCM_x, desiredTileSizeCM_y, desiredPESizeCM_x, desiredPESizeCM_y, CMTileheight, CMTilewidth, NMTileheight, NMTilewidth,
					&layerReadLatency, &layerReadDynamicEnergy, &tileLeakage, &layerbufferLatency, &layerbufferDynamicEnergy, &layericLatency, &layericDynamicEnergy,
					&coreLatencyADC, &coreLatencyAccum, &coreLatencyOther, &coreLatencyOther_only, &coreEnergyOther_only, &coreEnergyADC, &coreEnergyAccum, &coreEnergyOther, &global_routinglatency, &test, &local_bufferlatency, &local_bufferenergy
				, &local_routinglatency, &local_routingenergy, &global_routingenergy, &global_bufferlatency, &global_bufferenergy);

		double numTileOtherLayer = 0;
		double layerLeakageEnergy = 0;
		for (int j=0; j<netStructure.size(); j++) {
			if (j != i) {
				numTileOtherLayer += numTileEachLayer[0][j] * numTileEachLayer[1][j];
			}
		}
		layerLeakageEnergy = numTileOtherLayer*layerReadLatency*tileLeakage;

		chipReadLatency += layerReadLatency;
		avg_layerReadLatency += layerReadLatency;
		chipReadDynamicEnergy += layerReadDynamicEnergy;
		chipLeakageEnergy += layerLeakageEnergy;
		chipLeakage += tileLeakage*numTileEachLayer[0][i] * numTileEachLayer[1][i];
		chipbufferLatency += layerbufferLatency;
		chipbufferReadDynamicEnergy += layerbufferDynamicEnergy;
		chipicLatency += layericLatency;
		chipicReadDynamicEnergy += layericDynamicEnergy;

		global_routinglatency_1+=global_routinglatency;
		local_routinglatency_1+=local_routinglatency;
		global_bufferlatency_1+=global_bufferlatency;
		local_bufferlatency_1+=local_bufferlatency;
		coreLatencyOther_only_1+=coreLatencyOther_only;

		global_bufferenergy_1 += global_bufferenergy;
		coreEnergyOther_only_1 +=coreEnergyOther_only;
		local_bufferenergy_1 +=local_bufferenergy;
		global_routingenergy_1 +=global_routingenergy;
		local_routingenergy_1 +=local_routingenergy;




		chipLatencyADC += coreLatencyADC;
		chipLatencyAccum += coreLatencyAccum;
		//max_glob_acc_lat = MAX(max_glob_acc_lat, Global_accum_est_lat);

		//cout<<"\n The new global accum latency to support the chip max addition is: "<< Global_accum_est_lat*1e9<< "ns" <<endl;
		//cout<<"\n The new maximum global accum latency to support the chip max addition is: "<< max_glob_acc_lat*1e9<< "ns" <<endl;

		//chipLatencyAccum += max_glob_acc_lat;
		chipLatencyOther += coreLatencyOther;
		chipEnergyADC += coreEnergyADC;
		chipEnergyAccum += coreEnergyAccum;
		//max_glob_acc_energy = MAX(max_glob_acc_energy, Global_accum_est_energy);
		//cout<<"\n The new global accum energy to support the chip max addition is: "<< Global_accum_est_energy*1e12<< "pJ" <<endl;
		//cout<<"\n The new maximum global accum energy to support the chip max addition is: "<< max_glob_acc_energy*1e12<< "pJ" <<endl;
		//chipEnergyAccum += max_glob_acc_energy;

		chipEnergyOther += coreEnergyOther;
		global_iclatency += global_bufferlatency;

		cout << "layer" << i+1 << "'s readLatency is: " << layerReadLatency*1e9 << "ns" << endl;
		cout << "layer" << i+1 << "'s readDynamicEnergy is: " << layerReadDynamicEnergy*1e12 << "pJ" << endl;
		cout << "layer" << i+1 << "'s leakagePower is: " << numTileEachLayer[0][i] * numTileEachLayer[1][i] * tileLeakage*1e6 << "uW" << endl;
		cout << "layer" << i+1 << "'s leakageEnergy is: " << layerLeakageEnergy*1e12 << "pJ" << endl;
		cout << "layer" << i+1 << "'s buffer latency is: " << layerbufferLatency*1e9 << "ns" << endl;
		cout << "layer" << i+1 << "'s buffer readDynamicEnergy is: " << layerbufferDynamicEnergy*1e12 << "pJ" << endl;
		cout << "layer" << i+1 << "'s Routing latency is: " << layericLatency*1e9 << "ns" << endl;
		cout << "layer" << i+1 << "'s Routing readDynamicEnergy is: " << layericDynamicEnergy*1e12 << "pJ" << endl;

		count+=1;
		tile_total+=test;


		cout << endl;
		cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
		cout << endl;
		cout << "----------- ADC (or S/As and precharger for SRAM) readLatency is : " << coreLatencyADC*1e9 << "ns" << endl;
		cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readLatency is : " << coreLatencyAccum*1e9 << "ns" << endl;
		cout << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readLatency is : " << coreLatencyOther*1e9 << "ns" << endl;
		cout << "----------- Other Peripheries Only (e.g. decoders, mux, switchmatrix, pooling and activation units) readLatency is : " << coreLatencyOther_only*1e9 << "ns" << endl;
		cout << "----------- ADC (or S/As and precharger for SRAM) readEnergy is : " << coreEnergyADC*1e12 << "pJ" << endl;
		cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readEnergy is : " << coreEnergyAccum*1e12 << "pJ" << endl;
		cout << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readEnergy is : " << coreEnergyOther*1e12 << "pJ" << endl;
		cout << endl;
		cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
		cout << endl;


	}

	//cout<<"Count is "<<count<<endl;
	//cout<<"The total layer read latency is : "<<avg_layerReadLatency<<endl;
	avg_layerReadLatency = avg_layerReadLatency/count;

	cout << "------------------------------ Summary --------------------------------" <<  endl;
	cout << endl;
	cout << "------------------------------ Area Summary --------------------------------" <<  endl;
	cout << endl;
	cout << "ChipArea : " << chipArea*1e12 << " um^2" << endl;
	myarea<<"Chip Area"<<","<<chipArea * 1e12<<","<<"um^2"<<endl;
	cout << "Total Routing Area on chip (Tile/PE local): " << chipAreaIC*1e12 << " um^2" << endl;
	myarea<<"Total Within Tile Routing Area"<<","<<chipAreaIC * 1e12<<","<<"um^2"<<endl;
	cout << "Total ADC (or S/As and precharger for SRAM) Area on chip : " << chipAreaADC*1e12 << " um^2" << endl;
	myarea<<"Total ADC (or S/As and precharger for SRAM) Area"<<","<<chipAreaADC * 1e12<<","<<"um^2"<<endl;
	cout << "Total Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) on chip : " << chipAreaAccum*1e12 << " um^2" << endl;
	myarea<<"Total Accumulation Area"<<","<<chipAreaAccum * 1e12<<","<<"um^2"<<endl;
	cout << "Other Peripheries (e.g. decoders, mux, switchmatrix and activation units) : " << chipAreaOther*1e12 << " um^2" << endl;
	myarea<<"Total Other Peripheries Area"<<","<<chipAreaOther * 1e12<<","<<"um^2"<<endl;
	//cout<<  "The total buffer area is: "<<(chipArealocalbuffer + chipAreaglobalbuffer)*1e12 << " um^2" << endl;
	cout<<  "The total buffer area within tile is: "<<(chipArealocalbuffer)*1e12 << " um^2" << endl;
	myarea<<"Total Buffer Area within the Tile"<<","<<chipArealocalbuffer * 1e12<<","<<"um^2"<<endl;
	//cout<<  "The total local buffer area is " << chipArealocalbuffer*1e12 << " um^2" << endl;
	//myarea<<"Total local Buffer Area"<<","<<chipArealocalbuffer * 1e12<<","<<"um^2"<<endl;
	//cout<<	"The total global buffer area is "<< chipAreaglobalbuffer*1e12 << " um^2" << endl;
	//cout<<	"The total global routing area is "<< chipAreaglobalIC*1e12 << " um^2" << endl;
	//cout<< 	"The total local routing area is "<< chipAreatileIC*1e12 << " um^2" << endl;
	//myarea<<"Total local Routing Area"<<","<<chipAreatileIC * 1e12<<","<<"um^2"<<endl;
	//cout<		"The total subarray (compute engine) only area is "<< chipAreasubarrayalone*1e12 << " um^2"<<endl;
	cout << "------------------------------ Area Summary --------------------------------" <<  endl;
	cout << endl;
	myarea.close();

	cout << "------------------------------ Latency Summary --------------------------------" <<  endl;
	std::ofstream mylat;
	mylat.open ("./Final_Results/Latency.csv");
	std::ofstream myenergy;
	myenergy.open ("./Final_Results/Energy.csv");
	cout << endl;
	cout<<  "The average tile latency of the chip is : "<<avg_layerReadLatency*1e9<<" ns" << endl;
	cout << "Chip total readLatency is: " << chipReadLatency*1e9 << " ns" << endl;
	mylat<<"Total readLatency"<<","<<chipReadLatency * 1e9<<","<<"ns"<<endl;
	cout << "Chip buffer readLatency is: " << chipbufferLatency*1e9 << " ns" << endl;
	mylat<<"Total Buffer Latency"<<","<<chipbufferLatency * 1e9<<","<<"ns"<<endl;
	cout << "Chip Routing readLatency is: " << chipicLatency*1e9 << " ns" << endl;
	mylat<<"Total Routing Latency"<<","<<chipicLatency * 1e9<<","<<"ns"<<endl;
	//cout << "Chip total global routing latency is : "<<global_routinglatency_1*1e9<<" ns"<<endl;
	//cout << "Chip total tile routing latency is : "<<local_routinglatency_1*1e9<<" ns"<<endl;
	//cout << "Chip total global buffer latency is : "<<global_bufferlatency_1*1e9<<" ns"<<endl;
	//cout << "Chip total local buffer latency is : "<<local_bufferlatency_1*1e9<<" ns"<<endl;
	//mylat<<"Total local buffer Latency"<<","<<local_bufferlatency_1 * 1e9<<","<<"ns"<<endl;
	//cout << "Chip total other peripheries only latency is : "<<coreLatencyOther*1e9<<" ns"<<endl;

	cout<<	endl;
	cout << "************************ Breakdown of Latency *************************" << endl;
	cout << endl;
	cout << "----------- ADC (or S/As and precharger for SRAM) readLatency is : " << chipLatencyADC*1e9 << " ns" << endl;
	mylat<<"Total ADC Latency"<<","<<chipLatencyADC * 1e9<<","<<"ns"<<endl;
	cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readLatency is : " << chipLatencyAccum*1e9 << " ns" << endl;
	mylat<<"Total Accumulation Latency"<<","<<chipLatencyAccum * 1e9<<","<<"ns"<<endl;
	cout << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readLatency is : " << chipLatencyOther*1e9 << " ns" << endl;
	mylat<<"Total Other Peripheries Latency"<<","<<chipLatencyOther * 1e9<<","<<"ns"<<endl;
	cout << "************************ Breakdown of Latency *************************" << endl;
	cout<<	endl;
	cout << "************************ Energy Summary*************************" << endl;
	cout << "Chip total readDynamicEnergy is: " << chipReadDynamicEnergy*1e12 << " pJ" << endl;
	myenergy<<"Total readEnergy"<<","<<chipReadDynamicEnergy * 1e12<<","<<"pJ"<<endl;
	cout << "Chip total leakage Energy is: " << chipLeakageEnergy*1e12 << " pJ" << endl;
	myenergy<<"Total leakage Energy"<<","<<chipLeakageEnergy * 1e12<<","<<"pJ"<<endl;
	cout << "Chip total leakage Power is: " << chipLeakage*1e6 << " uW" << endl;
	myenergy<<"Total leakage Power"<<","<<chipLeakage * 1e6<<","<<"uW"<<endl;
	cout << "Chip Routing readDynamicEnergy is: " << chipicReadDynamicEnergy*1e12 << " pJ" << endl;
	myenergy<<"Total Routing Energy"<<","<<chipicReadDynamicEnergy * 1e12<<","<<"pJ"<<endl;
	cout << "Chip Buffer readDynamicEnergy is: " << chipbufferReadDynamicEnergy*1e12 << " pJ" << endl;
	myenergy<<"Total Buffer Energy"<<","<<chipbufferReadDynamicEnergy * 1e12<<","<<"pJ"<<endl;
	//cout << "Chip total other peripheries energy is : "<<coreEnergyOther_only_1*1e12<<" pJ"<<endl;
	cout<<endl;
	//cout << "Chip Global buffer readDynamicEnergy is: " << global_bufferenergy_1*1e12 << " pJ" << endl;
	cout << "Chip Local buffer readDynamicEnergy is: " << local_bufferenergy_1*1e12 << " pJ" << endl;
	myenergy<<"Total Local Buffer Energy"<<","<<local_bufferenergy_1 * 1e12<<","<<"pJ"<<endl;
	//cout << "Chip Global Routing readDynamicEnergy is: " << global_routingenergy_1*1e12 << " pJ" << endl;
	cout << "Chip Local Routing readDynamicEnergy is: " << local_routingenergy_1*1e12 << " pJ" << endl;
	myenergy<<"Total Local Routing Energy"<<","<<local_routingenergy_1 * 1e12<<","<<"pJ"<<endl;
	cout<<endl;

	cout << endl;
	cout << "************************ Breakdown of Dynamic Energy *************************" << endl;
	cout << endl;
	//cout << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, pooling and activation units) readLatency is : " << coreLatencyOther*1e9 << " ns" << endl;
	cout << "----------- ADC (or S/As and precharger for SRAM) readEnergy is : " << chipEnergyADC*1e12 << " pJ" << endl;
	myenergy<<"Total ADC Energy"<<","<<chipEnergyADC * 1e12<<","<<"pJ"<<endl;
	cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readEnergy is : " << chipEnergyAccum*1e12 << " pJ" << endl;
	myenergy<<"Total Accumulation Energy"<<","<<chipEnergyAccum * 1e12<<","<<"pJ"<<endl;
	cout << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readEnergy is : " << chipEnergyOther*1e12 << " pJ" << endl;
	myenergy<<"Total Other Peripheries Energy"<<","<<chipEnergyOther * 1e12<<","<<"pJ"<<endl;
	//cout << "----------- Other Peripheries Only (e.g. decoders, mux, switchmatrix, pooling and activation units) readEnergy is : " << coreEnergyOther_only_1*1e12 << " pJ" << endl;
	cout << endl;
	cout << "************************ Breakdown Dynamic Energy *************************" << endl;
	cout << endl;
	mylat.close();
	cout << endl;
	//cout << "----------------------------- Performance -------------------------------" << endl;
	//cout << "Energy Efficiency TOPS/W (Layer-by-Layer Process): " << numComputation/(chipReadDynamicEnergy*1e12+chipLeakageEnergy*1e12) << endl;
	//myenergy<<"Energy Efficiency TOPS/W "<<","<<numComputation/(chipReadDynamicEnergy*1e12+chipLeakageEnergy*1e12)<<endl;
	//cout << "Throughput FPS (Layer-by-Layer Process): " << 1/(chipReadLatency) << endl;
	//myenergy<<"Throughput "<<","<<1/(chipReadLatency)<<endl;
	myenergy.close();
	std::ofstream myfile_1;
	myfile_1.open ("./to_interconnect/fps.csv");
	myfile_1 << 1/(chipReadLatency) <<endl;
	myfile_1.close();
	cout << "-------------------------------------- Hardware Performance Done --------------------------------------" <<  endl;
	cout << endl;
	auto stop = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::seconds>(stop-start);
  	cout << "------------------------------ Simulation Performance --------------------------------" <<  endl;
	cout << "Total Run-time of SIAM: " << duration.count() << " seconds" << endl;
	cout << "------------------------------ Simulation Performance --------------------------------" <<  endl;

	return 0;

}

vector<vector<double> > getNetStructure(const string &inputfile) {
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

	vector<vector<double> > netStructure;
	for (int row=0; row<ROWin; row++) {
		vector<double> netStructurerow;
		getline(infile, inputline, '\n');
		istringstream iss;
		iss.str(inputline);
		for (int col=0; col<COLin; col++) {
			while(getline(iss, inputval, ',')){
				istringstream fs;
				fs.str(inputval);
				double f=0;
				fs >> f;
				netStructurerow.push_back(f);
			}
		}
		netStructure.push_back(netStructurerow);
	}
	infile.close();
	//cout<<"The size of netStructure is:"<<netStructure.size();
	return netStructure;
	netStructure.clear();
}
