#include "robometrics/evaluation.hpp"
#include <cmath>
#include <stdexcept>
using namespace robometrics;
int main() {
    Matrix identity={1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
    Model m; m.base=m.ee_offset=identity; m.origins={identity,identity,identity};
    m.origins[2][3]=1;
    m.parents={-1,0,1}; m.q_indices={-1,0,-1}; m.axes={{0,0,1},{0,0,1},{0,0,1}};
    m.ee_link=2; m.sphere_centers={{0,0,0}}; m.radii={.1f}; m.sphere_links={2};
    m.padding={0,0,0}; m.lower={-4}; m.upper={4}; m.velocity={1};
    Evaluator evaluator(m);
    float q=1.57079632679f;
    const auto states=evaluator.states(&q,1,nullptr,nullptr);
    if(std::abs(states[0].ee[0])>1e-6f || std::abs(states[0].ee[1]-1)>1e-6f) return 1;
    float grid[8]={0,1,2,3,4,5,6,7};
    EsdfView map; map.grid=grid; map.shape={2,2,2}; map.origin={.5,.5,.5};
    map.lower={0,0,0}; map.upper={2,2,2}; map.voxel=1;
    if(std::abs(map.query({1,1,1})-3.5f)>1e-6) return 2;
    if(!map.valid({0,0,0}) || map.valid({2,1,1}) || map.query({2,1,1})!=0) return 3;
    q=0;
    float velocity=2, acceleration=17, jerk=501;
    Pose goal=evaluator.states(&q,1,nullptr,nullptr)[0].ee;
    const auto metrics=evaluator.trajectory(&q,&velocity,&acceleration,&jerk,1,goal,map,{15,500});
    if(metrics.max_velocity_violation!=1 || metrics.max_acceleration_violation!=2 || metrics.max_abs_jerk!=501) return 5;
    m.parents[1]=2;
    try { Evaluator invalid(m); } catch(const std::invalid_argument&) { return 0; }
    return 4;
}
