#include <iostream>
#include <memory>

#include <eigen3/Eigen/Dense>
#include <se3.hpp>

#include <ceres/ceres.h>
#include <glog/logging.h>

struct Frame {
    Eigen::Vector2f *gradient;
    float *gray;
    float *depth;
    int width;
    int height;
    float fx;
    float fy;
    float cx;
    float cy;
};

typedef Eigen::Matrix<float, 1, 6> Jaccobi;


class QuadraticCostFunction : public ceres::SizedCostFunction<1, 1, 1, 1, 1, 1, 1> {
public:
    QuadraticCostFunction() {}
    void setParam(std::shared_ptr<Frame> &framei, std::shared_ptr<Frame> &framej, int uj, int vj) {
        frame_i = framei;
        frame_j = framej;
        this->uj = uj;
        this->vj = vj;
    }

    virtual bool Evaluate(double const* const* parameters,
                            double* residuals,
                            double** jacobians) const {
        Eigen::Matrix<float, 6, 1> vec(parameters[0][0], parameters[0][1], parameters[0][2],
                parameters[0][3], parameters[0][4], parameters[0][5]);

        Sophus::SE3f pose = Sophus::SE3f::exp(vec);

        RetType ret = calc(frame_i, frame_j, pose, uj, vj);

        residuals = ret.err;

        if (jacobians != NULL && jacobians[0] != NULL) {
            jacobians[0][0] = ret.jac(0);
            jacobians[0][1] = ret.jac(1);
            jacobians[0][2] = ret.jac(2);
            jacobians[0][3] = ret.jac(3);
            jacobians[0][4] = ret.jac(4);
            jacobians[0][5] = ret.jac(5);
        }

        return true;
    }

private:
    struct RetType {
        Jaccobi jac;
        float   err;
    };

    RetType calc(std::shared_ptr<Frame> frame_i, std::shared_ptr<Frame> frame_j, Sophus::SE3f &pose ,int uj, int vj) {
        RetType ret;
        float depth = frame_j->depth[vj * frame_j->width + uj];
        Eigen::Vector3f p(depth * uj, depth * vj, depth);
        p(0) = p(0) / frame_j->fx - frame_j->cx / frame_j->fx;
        p(1) = p(1) / frame_j->fy - frame_j->cy / frame_j->fy;
        Eigen::Vector3f p_ = pose.rotationMatrix() * p + pose.translation();
        int ui = p_(0) / p_(2);
        int vi = p_(1) / p_(2);

        ret.jac(0) = frame_j->fx / depth * frame_i->gradient[vi * frame_i->width + ui](0);
        ret.jac(1) = frame_j->fy / depth * frame_i->gradient[vi * frame_i->width + ui](1);
        ret.jac(2) = -frame_j->fx * p(0) / depth / depth * frame_i->gradient[vi * frame_i->width + ui](0)
                - frame_j->fy * p(1) / depth / depth * frame_i->gradient[vi * frame_i->width + ui](1);

        ret.jac(3) =  - p(0) * p(1) / depth / depth * frame_j->fx * frame_i->gradient[vi * frame_i->width + ui](0)
                - (1 + p(1) * p(1) / p(2) / p(2)) * frame_j->fy * frame_i->gradient[vi * frame_i->width + ui](1);

        ret.jac(4) = (1 + p(0) * p(0) / p(2) / p(2)) * frame_j->fx * frame_i->gradient[vi * frame_i->width + ui](0)
                + p(0) * p(1) / depth / depth * frame_j->fy * frame_i->gradient[vi * frame_i->width + ui](1);

        ret.jac(5) = -frame_j->fx * p(1) / depth * frame_i->gradient[vi * frame_i->width + ui](0)
                 + frame_j->fy * p(0) / depth * frame_i->gradient[vi * frame_i->width + ui](1);

        ret.err = frame_i->gray[vi * frame_i->width + ui] - frame_j->gray[vj * frame_j->width + uj];

        return ret;
    }


private:
    std::shared_ptr<Frame> frame_i;
    std::shared_ptr<Frame> frame_j;
    int uj, vj;
};

int main(int argc, char* argv[])
{
    google::InitGoogleLogging(argv[0]);
    ceres::Problem *problem = new ceres::Problem;
    Sophus::SE3f initPose;
    int trackingNum = 50;
    int uj[50];
    int vj[50];
    std::shared_ptr<Frame> frame_i, frame_j;
    Eigen::Matrix<float, 6, 1> se3Pose = initPose.log();
    QuadraticCostFunction *costFunc = new QuadraticCostFunction[trackingNum];

    for(int i = 0; i < 50; ++i) {
        costFunc[i].setParam(frame_i, frame_j, uj[i], vj[i]);
        problem->AddResidualBlock(&costFunc[i], &se3Pose[0], &se3Pose[1],
                &se3Pose[2], &se3Pose[3], &se3Pose[4], &se3Pose[5]);
    }

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    ceres::Solve(options, problem, &summary);
//    std::cout << summary.BriefReport() << std::endl;

    Sophus::SE3f Pose = Sophus::SE3f::exp(se3Pose);

    return 0;
}

