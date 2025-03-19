/**
 * Fast map matching.
 *
 * Definition of a TransitionGraph.
 *
 * @author: Can Yang
 * @version: 2020.01.31
 */

#include "mm/transition_graph.hpp"
#include "network/type.hpp"
#include "util/debug.hpp"

using namespace FMM;
using namespace FMM::CORE;
using namespace FMM::NETWORK;
using namespace FMM::MM;

TransitionGraph::TransitionGraph(const Traj_Candidates &tc, double gps_error, bool perturbation){
  for (auto cs = tc.begin(); cs!=tc.end(); ++cs) {
    TGLayer layer;
    for (auto iter = cs->begin(); iter!=cs->end(); ++iter) {
      double ep = calc_ep(iter->dist,gps_error,perturbation);
      layer.push_back(TGNode{&(*iter),nullptr,ep,0,
        -std::numeric_limits<double>::infinity(),0});
    }
    layers.push_back(layer);
  }
  if (!tc.empty()) {
    reset_layer(&(layers[0]));
  }
}

double TransitionGraph::calc_tp(double sp_dist,double eu_dist){
  return eu_dist>=sp_dist ? 1.0 : eu_dist/sp_dist;
}

double TransitionGraph::calc_ep(double dist,double error,bool perturbation) {
  if (perturbation) {
    double dist_2 = dist * dist;       
    double dist_3 = dist_2 * dist;      
    double dist_4 = dist_3 * dist;
    // normalization factor of distribution
    switch(static_cast<int>(error)) {
      case 20:
        if(dist < 0.00018) {
          return (8.939 * 1e14 * dist_4) 
              - ( 2.232 * 1e11 * dist_3)
              - ( 5.173 * 1e7 * dist_2)
              + ( 1.316 * 1e4 * dist) + 0.3349;
        }
        else {
          return 1.455 *1e4 * dist_2 - 66.83 * dist + 0.08811;
        }
 
      case 30:
        if(dist < 0.000275) {
          return (-3.037 * 1e14 * dist_4) 
              + (1.819 * 1e11 * dist_3)
              - ( 6.54 * 1e7 * dist_2)
              + (1.154 * 1e4 * dist) + 0.2797;
        }
        else {
          return 1.49 * 1e4 * dist_2 - 78.11 * dist + 0.1159;
        }

      case 40:
        if(dist < 0.000365) {
          return (-4.823 * 1e13 * dist_4) 
              + (4.507 * 1e10 * dist_3)
              - (3.035 * 1e7 * dist_2)
              + (8302 * dist) + 0.2701;
        }
        else {
          return  1.993 * 1e4 * dist_2 - 104.1 * dist + 0.1535;
        }

      case 50:
        if(dist < 0.000458) {
          return (-6.547 * 1e13 * dist_4) 
              + (6.186 * 1e10 * dist_3)
              - (3.003 * 1e7 * dist_2)
              + (7729 * dist) +0.232;
        }
        else {
          return 2.474 * 1e4 * dist_2 - 129.7 * dist + 0.1916;
        }

      case 60:
        if(dist < 0.00055) {
          return (-3.286 * 1e13 * dist_4) 
              + (3.626 * 1e10 * dist_3)
              - (2.075 * 1e7 * dist_2)
              + (6419 * dist) + 0.2268;
        }
        else {
          return 3.023 * 1e4 * dist_2 - 157.6 * dist + 0.2312;
        }

      case 70:
        if(dist < 0.00064) {
          return (-2.031 * 1e13 * dist_4) 
              + (2.534 * 1e10 * dist_3)
              - (1.571 * 1e7 * dist_2)
              + (5517 * dist) + 0.2054;
        }
        else {
          return 3.596 * 1e4 * dist_2 - 187.1 * dist + 0.2732;
        }
 
      case 80:
        if(dist < 0.000733) {
          return (-1.606 * 1e13 * dist_4) 
              + (2.312 * 1e10 * dist_3)
              - (1.483 * 1e7 * dist_2)
              + (5244 * dist) + 0.1954;
        }
        else {
          return 4.16 * 1e4 * dist_2 - 217.6 * dist + 0.3187;
        }

      case 90:
        if(dist < 0.000825) {
          return (-8.875 * 1e12 * dist_4) 
              + (1.455 * 1e10 * dist_3)
              - (1.109 * 1e7 * dist_2)
              + (4635 * dist) + 0.1909;
        }
        else {
          return 4.671 * 1e4 * dist_2 - 246.6 * dist + 0.3631;
        }

      case 100:
        if(dist < 0.000915) {
          return (-4.13 * 1e12 * dist_4) 
              + (6.907 * 1e9 * dist_3)
              - (6.353 * 1e6 * dist_2)
              + (3558 * dist) + 0.1925;
        }
        else {
          return 5.184 * 1e4 * dist_2 - 274.8 * dist + 0.4052;
        }

      default:
        if(dist < 0.000458) {
          return (-6.547 * 1e13 * dist_4) 
              + (6.186 * 1e10 * dist_3)
              - (3.003 * 1e7 * dist_2)
              + (7729 * dist) +0.232;
        }
        else {
          return 2.474 * 1e4 * dist_2 - 129.7 * dist + 0.1916;
        }
    }
  }
  else {
    double a = dist / error;
    return exp(-0.5 * a * a); 
  } 
}

// Reset the properties of a candidate set
void TransitionGraph::reset_layer(TGLayer *layer){
  for (auto iter=layer->begin(); iter!=layer->end(); ++iter) {
    iter->cumu_prob = log(iter->ep);
    iter->prev = nullptr;
  }
}

const TGNode *TransitionGraph::find_optimal_candidate(const TGLayer &layer){
  const TGNode *opt_c=nullptr;
  double final_prob = -std::numeric_limits<double>::infinity();
  for (auto c = layer.begin(); c!=layer.end(); ++c) {
    if(final_prob < c->cumu_prob) {
      final_prob = c->cumu_prob;
      opt_c = &(*c);
    }
  }
  return opt_c;
}

TGOpath TransitionGraph::backtrack(){
  SPDLOG_TRACE("Backtrack on transition graph");
  TGNode* track_cand=nullptr;
  double final_prob = -std::numeric_limits<double>::infinity();
  std::vector<TGNode>& last_layer = layers.back();
  for (auto c = last_layer.begin(); c!=last_layer.end(); ++c) {
    if(final_prob < c->cumu_prob) {
      final_prob = c->cumu_prob;
      track_cand = &(*c);
    }
  }
  TGOpath opath;
  int i = layers.size();
  if (final_prob>-std::numeric_limits<double>::infinity()) {
    opath.push_back(track_cand);
    --i;
    SPDLOG_TRACE("Optimal candidate {} edge id {} sp {} tp {} cp {}",
        i,track_cand->c->edge->id,track_cand->sp_dist,track_cand->tp,
        track_cand->cumu_prob);
    // Iterate from tail to head to assign path
    while ((track_cand=track_cand->prev)!=nullptr) {
      opath.push_back(track_cand);
      --i;
      SPDLOG_TRACE("Optimal candidate {} edge id {} sp {} tp {} cp {}",
        i,track_cand->c->edge->id,track_cand->sp_dist,track_cand->tp,
        track_cand->cumu_prob);
    }
    std::reverse(opath.begin(), opath.end());
  }
  SPDLOG_TRACE("Backtrack on transition graph done");
  return opath;
}

void TransitionGraph::print_optimal_info(){
  int N = layers.size();
  if (N<1) return;
  const TGNode *global_opt_node = nullptr;
  for (int i=N-1;i>=0;--i){
    const TGNode *local_opt_node = find_optimal_candidate(layers[i]);
    if (global_opt_node!=nullptr){
      global_opt_node=global_opt_node->prev;
    } else {
      global_opt_node=local_opt_node;
    }
    SPDLOG_TRACE("Point {} global opt {} local opt {}",
      i, (global_opt_node==nullptr)?-1:global_opt_node->c->edge->id,
         (local_opt_node==nullptr)?-1:local_opt_node->c->edge->id);
  }
};

std::vector<TGLayer> &TransitionGraph::get_layers(){
  return layers;
}
