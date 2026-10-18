#include "robometrics/evaluation.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace robometrics {
namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
template<class Array> bool finite(const Array& values) {
    for (auto v : values) if (!std::isfinite(v)) return false;
    return true;
}
float norm(const Vec3& v) { return std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); }
Matrix multiply(const Matrix& a, const Matrix& b) {
    Matrix result{};
    for (int r=0; r<4; ++r) for (int c=0; c<4; ++c)
        for (int k=0; k<4; ++k) result[r*4+c] += a[r*4+k]*b[k*4+c];
    return result;
}
Vec3 transform(const Matrix& t, const Vec3& p) {
    Vec3 result{};
    for (int r=0; r<3; ++r) {
        for (int k=0; k<3; ++k) result[r] += t[r*4+k]*p[k];
        result[r] += t[r*4+3];
    }
    return result;
}
Matrix rotation(const Vec3& axis, float angle) {
    const float c=std::cos(angle), s=std::sin(angle), v=1-c;
    const float x=axis[0], y=axis[1], z=axis[2];
    return {c+x*x*v, x*y*v-z*s, x*z*v+y*s, 0,
            y*x*v+z*s, c+y*y*v, y*z*v-x*s, 0,
            z*x*v-y*s, z*y*v+x*s, c+z*z*v, 0,
            0,0,0,1};
}
Pose pose(const Matrix& t) {
    const float a=t[0],b=t[1],c=t[2],d=t[4],e=t[5],f=t[6],g=t[8],h=t[9],i=t[10];
    const float scores[4]={1+a+e+i,1+a-e-i,1-a+e-i,1-a-e+i};
    int best=0;
    for (int k=1;k<4;++k) if (scores[k]>scores[best]) best=k;
    const std::array<float,4> candidates[4]={
        {h-f,c-g,d-b,scores[0]}, {scores[1],b+d,c+g,h-f},
        {b+d,scores[2],f+h,c-g}, {c+g,f+h,scores[3],d-b}};
    auto q=candidates[best];
    float length=0;
    for (auto value:q) length+=value*value;
    length=std::sqrt(length);
    const float sign=q[3]<0 ? -1.f : 1.f;
    return {t[3],t[7],t[11],sign*q[0]/length,sign*q[1]/length,sign*q[2]/length,sign*q[3]/length};
}
void rigid(const Matrix& t) {
    require(finite(t), "transforms must be finite");
    require(t[12]==0 && t[13]==0 && t[14]==0 && t[15]==1, "invalid homogeneous transform");
    for (int a=0;a<3;++a) for(int b=0;b<3;++b) {
        float dot=0;
        for(int k=0;k<3;++k) dot+=t[k*4+a]*t[k*4+b];
        require(std::abs(dot-(a==b ? 1.f:0.f))<1e-4f,"rotation must be orthonormal");
    }
    const float det=t[0]*(t[5]*t[10]-t[6]*t[9])-t[1]*(t[4]*t[10]-t[6]*t[8])+t[2]*(t[4]*t[9]-t[5]*t[8]);
    require(std::abs(det-1)<1e-4f,"rotation must have determinant +1");
}
}
void Model::validate() const {
    const auto n=origins.size(), dof=lower.size();
    require(n>0 && dof>0,"model requires links and degrees of freedom");
    require(parents.size()==n && q_indices.size()==n && axes.size()==n && padding.size()==n,"link field sizes disagree");
    require(upper.size()==dof && velocity.size()==dof,"joint field sizes disagree");
    require(ee_link>=0 && static_cast<std::size_t>(ee_link)<n,"invalid EE link");
    rigid(base); rigid(ee_offset);
    require(parents[0]==-1 && q_indices[0]==-1,"link zero must be the fixed base");
    std::vector<int> used(dof,0);
    for (std::size_t link=0;link<n;++link) {
        rigid(origins[link]);
        require(link==0 || (parents[link]>=0 && static_cast<std::size_t>(parents[link])<link),"links must be topologically ordered");
        const auto index=q_indices[link];
        require(index>=-1 && index<static_cast<int>(dof),"invalid q index");
        require(finite(axes[link]),"axis must be finite");
        if(index>=0) {
            require(std::abs(norm(axes[link])-1)<1e-5f,"revolute axis must be unit length");
            used[index]++;
        }
        require(std::isfinite(padding[link]) && padding[link]>=0,"padding must be finite and nonnegative");
    }
    for(std::size_t j=0;j<dof;++j) {
        require(used[j]==1,"each DOF must occur exactly once");
        require(std::isfinite(lower[j]) && std::isfinite(upper[j]) && lower[j]<upper[j],"invalid joint bounds");
        require(std::isfinite(velocity[j]) && velocity[j]>0,"invalid velocity limit");
    }
    require(!radii.empty() && radii.size()==sphere_centers.size() && radii.size()==sphere_links.size(),"invalid sphere arrays");
    for(std::size_t k=0;k<radii.size();++k) {
        require(std::isfinite(radii[k]) && radii[k]>0 && finite(sphere_centers[k]),"invalid sphere");
        require(sphere_links[k]>=0 && static_cast<std::size_t>(sphere_links[k])<n,"invalid sphere link");
    }
    for(auto pair:pairs) require(pair[0]>=0 && pair[1]>pair[0] && static_cast<std::size_t>(pair[1])<radii.size(),"invalid sphere pair");
}
void EsdfView::validate() const {
    require(grid!=nullptr && std::isfinite(voxel) && voxel>0,"invalid map storage or voxel size");
    require(finite(origin) && finite(lower) && finite(upper),"map coordinates must be finite");
    for(int k=0;k<3;++k) require(shape[k]>=2 && lower[k]<upper[k],"invalid map dimensions or bounds");
}
bool EsdfView::valid(const Vec3& p) const {
    for(int k=0;k<3;++k) if(!(p[k]>=lower[k] && p[k]<upper[k])) return false;
    return true;
}
float EsdfView::query(const Vec3& p) const {
    int lo[3], hi[3]; float d[3];
    for(int k=0;k<3;++k) {
        float u=(p[k]-origin[k])/voxel;
        if(!(u>=-.5f && u<static_cast<float>(shape[k])-.5f)) return 0;
        u=std::clamp(u,0.f,static_cast<float>(shape[k]-1));
        lo[k]=static_cast<int>(std::floor(u)); hi[k]=std::min(lo[k]+1,static_cast<int>(shape[k]-1));
        d[k]=u-static_cast<float>(lo[k]);
    }
    auto at=[&](int x,int y,int z){ return grid[(x*shape[1]+y)*shape[2]+z]; };
    float c00=at(lo[0],lo[1],lo[2])*(1-d[0])+at(hi[0],lo[1],lo[2])*d[0];
    float c01=at(lo[0],lo[1],hi[2])*(1-d[0])+at(hi[0],lo[1],hi[2])*d[0];
    float c10=at(lo[0],hi[1],lo[2])*(1-d[0])+at(hi[0],hi[1],lo[2])*d[0];
    float c11=at(lo[0],hi[1],hi[2])*(1-d[0])+at(hi[0],hi[1],hi[2])*d[0];
    return (c00*(1-d[1])+c10*d[1])*(1-d[2])+(c01*(1-d[1])+c11*d[1])*d[2];
}
Evaluator::Evaluator(Model model): model_(std::move(model)) {
    model_.validate(); poses_.resize(model_.origins.size()); spheres_.resize(model_.radii.size());
}
StateMetrics Evaluator::state(const float* q,const Pose* goal,const EsdfView* esdf) {
    poses_[0]=model_.base;
    for(std::size_t k=1;k<poses_.size();++k) {
        poses_[k]=multiply(poses_[model_.parents[k]],model_.origins[k]);
        if(model_.q_indices[k]>=0) poses_[k]=multiply(poses_[k],rotation(model_.axes[k],q[model_.q_indices[k]]));
    }
    StateMetrics result;
    result.ee=pose(multiply(poses_[model_.ee_link],model_.ee_offset));
    if(goal) {
        Vec3 delta{};
        for(int k=0;k<3;++k) delta[k]=result.ee[k]-(*goal)[k];
        result.position_error=norm(delta);
        float dot=0;
        for(int k=3;k<7;++k) dot+=result.ee[k]*(*goal)[k];
        result.orientation_error=1-std::min(1.f,std::abs(dot));
    }
    for(std::size_t j=0;j<model_.lower.size();++j) {
        const float gap=std::max({model_.lower[j]-q[j],q[j]-model_.upper[j],0.f});
        result.max_joint_violation=std::max(result.max_joint_violation,gap);
        result.normalized_joint_violation=std::max(result.normalized_joint_violation,gap/(model_.upper[j]-model_.lower[j]));
    }
    if(!esdf) return result;
    for(std::size_t k=0;k<spheres_.size();++k) {
        spheres_[k]=transform(poses_[model_.sphere_links[k]],model_.sphere_centers[k]);
        if(esdf->valid(spheres_[k])) {
            ++result.valid_sphere_count;
            result.min_clearance=std::min(result.min_clearance,esdf->query(spheres_[k])-model_.radii[k]);
        } else ++result.oob_count;
    }
    for(auto pair:model_.pairs) {
        const auto a=pair[0], b=pair[1]; Vec3 delta{};
        for(int k=0;k<3;++k) delta[k]=spheres_[a][k]-spheres_[b][k];
        const float required=model_.radii[a]+model_.radii[b]+model_.padding[model_.sphere_links[a]]+model_.padding[model_.sphere_links[b]];
        result.min_self_clearance=std::min(result.min_self_clearance,norm(delta)-required);
    }
    return result;
}
std::vector<StateMetrics> Evaluator::states(const float* q,std::size_t count,const Pose* goal,const EsdfView* esdf) {
    if(esdf) esdf->validate();
    require(q || count==0,"null state storage");
    if(goal) require(finite(*goal),"goal must be finite");
    const auto dof=model_.lower.size();
    for(std::size_t i=0;i<count*dof;++i) require(std::isfinite(q[i]),"states must be finite");
    std::vector<StateMetrics> result; result.reserve(count);
    for(std::size_t i=0;i<count;++i) result.push_back(state(q+i*dof,goal,esdf));
    return result;
}
TrajectoryMetrics Evaluator::trajectory(const float* q,const float* dq,const float* ddq,
                                        const float* jerk,std::size_t count,const Pose& goal,
                                        const EsdfView& esdf,DynamicsLimits limits) {
    require(count>0 && dq && ddq && jerk,"trajectory arrays must be nonempty");
    require(std::isfinite(limits.acceleration) && limits.acceleration>0 &&
            std::isfinite(limits.jerk) && limits.jerk>0,"dynamics limits must be finite and positive");
    auto profile=states(q,count,&goal,&esdf);
    TrajectoryMetrics result; result.ee.reserve(count);
    for(const auto& value:profile) {
        result.ee.push_back(value.ee);
        result.min_clearance=std::min(result.min_clearance,value.min_clearance);
        result.min_self_clearance=std::min(result.min_self_clearance,value.min_self_clearance);
        result.oob_count+=value.oob_count;
        result.max_joint_violation=std::max(result.max_joint_violation,value.max_joint_violation);
    }
    result.position_error=profile.back().position_error;
    result.orientation_error=profile.back().orientation_error;
    const auto dof=model_.lower.size();
    for(std::size_t i=0;i<count*dof;++i) {
        require(std::isfinite(dq[i]) && std::isfinite(ddq[i]) && std::isfinite(jerk[i]),"trajectory derivatives must be finite");
        result.max_velocity_violation=std::max(result.max_velocity_violation,std::abs(dq[i])-model_.velocity[i%dof]);
        result.max_acceleration_violation=std::max(result.max_acceleration_violation,std::abs(ddq[i])-limits.acceleration);
        result.max_abs_jerk=std::max(result.max_abs_jerk,std::abs(jerk[i]));
    }
    return result;
}
} // namespace robometrics
