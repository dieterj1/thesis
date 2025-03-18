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
    const double meter_per_degree = 109662.80313373724;

    switch(static_cast<int>(std::round(error*meter_per_degree))) {
      case 20:
      if(dist < 0.00018) {
        return (6.273 * 1e15 * dist_4) 
            - ( 2.03 * 1e12 * dist_3)
            + (1.041 * 1e8 * dist_2)
            + ( 1.193 * 1e4 * dist) + 0.151;
      }
      else {
        return 9910 * dist_2 - 48.93 * dist + 0.0694;
      }

      case 30:
      if(dist < 0.000275) {
        return (9.98 * 1e14 * dist_4) 
            - (4.466 * 1e11 * dist_3)
            + ( 1.093 * 1e7 * dist_2)
            + (1.081 * 1e4 * dist) + 0.1053;
      }
      else {
        return 9964 * dist_2 - 56.9 * dist + 0.09053;
      }

      case 40:
      if(dist < 0.000365) {
        return (2.287 * 1e14 * dist_4) 
            - (1.214 * 1e11 * dist_3)
            - (1.107 * 1e7 * dist_2)
            + (9712 * dist) + 0.07391;
      }
      else {
        return 1.31 * 1e4 * dist_2 - 74.95 * dist + 0.1192;
      }

      case 50:
      if(dist < 0.000458) {
        return (1.459 * 1e14 * dist_4) 
            - (1.148 * 1e11 * dist_3)
            + (1.003 * 1e7 * dist_2)
            + (5894 * dist) +0.09841;
      }
      else {
        return 1.617 * 1e4 * dist_2 - 91.67 * dist + 0.1448;
      }

      case 60:
      if(dist < 0.00055) {
        return (5.087 * 1e13 * dist_4) 
            - (4.303 * 1e10 * dist_3)
            - (2.264 * 1e6 * dist_2)
            + (6131 * dist) + 0.08209;
      }
      else {
        return 1.929 * 1e4 * dist_2 - 109.7 * dist + 0.1735;
      }

      case 70:
      if(dist < 0.00064) {
        return (3.084 * 1e13 * dist_4) 
            - (3.315 * 1e10 * dist_3)
            + (1.709 * 1e6 * dist_2)
            + (4658 * dist) + 0.0942;
      }
      else {
        return 2.293 * 1e4 * dist_2 - 129.1 * dist + 0.2026;
      }
  
      case 80:
      if(dist < 0.000733) {
        return (1.654 * 1e13 * dist_4) 
            - (1.913 * 1e10 * dist_3)
            - (5.956 * 1e5 * dist_2)
            + (4440 * dist) + 0.08997;
      }
      else {
        return 2.62 * 1e4 * dist_2 - 147.7 * dist + 0.2321;
      }

      case 90:
      if(dist < 0.000826) {
        return (1.337 * 1e13 * dist_4) 
            - (1.976 * 1e10 * dist_3)
            + (3.702 * 1e6 * dist_2)
            + (3081 * dist) + 0.09764;
      }
      else {
        return 2.869 * 1e4 * dist_2 - 162.6 * dist + 0.2568;
      }

      case 100:
      if(dist < 0.000915) {
        return - (1.659 * 1e12 * dist_4) 
            + (5.471 * 1e9 * dist_3)
            - (9.124 * 1e6 * dist_2)
            + (5248 * dist) + 0.05154;
      }
      else {
        return 3.436 * 1e4 * dist_2 - 195.1 * dist + 0.308;
      }

      default:
        double a = dist / error;
        return exp(-0.5 * a * a); 

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
