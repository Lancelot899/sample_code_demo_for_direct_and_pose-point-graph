#include <iostream>

#include <Eigen/Dense>
#include <se3.hpp>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>

class VertexPose : public g2o::BaseVertex<6, Sophus::SE3f> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void setSE3(Sophus::SE3f &se3) {
        setEstimate(se3);
    }

    virtual void setToOriginImpl()
    {
        _estimate = Sophus::SE3f();
    }

    bool read(std::istream &) {return true;}
    bool write(std::ostream &) const {return true;}

    virtual void oplusImpl ( const double* update ) {
        Sophus::SE3f up (
                    Sophus::SO3f ( update[3], update[4], update[5] ),
                Eigen::Vector3f ( update[0], update[1], update[2] ));
        _estimate = up * _estimate;
    }
};

class VertexPoint : public g2o::BaseVertex<3, Eigen::Vector3f> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void setPoint(Eigen::Vector3f &point) {
        setEstimate(point);
    }

    virtual void setToOriginImpl()
    {
        _estimate = Eigen::Vector3f();
    }

    bool read(std::istream &) {return true;}
    bool write(std::ostream &) const {return true;}

    virtual void oplusImpl ( const double* update ) {
        for(int i = 0; i < 3; ++i)
            _estimate(i) = update[i] + _estimate(i);
    }
};

class Edge : public g2o::BaseBinaryEdge<2, Eigen::Vector2f, VertexPose, VertexPoint> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void setEdge(float fx, float fy, float cx, float cy, Eigen::Vector2f &Point) {
        setMeasurement(Point);
        this->fx = fx;
        this->fy = fy;
        this->cx = cx;
        this->cy = cy;
    }

    bool read(std::istream &) {return true;}
    bool write(std::ostream &) const {return true;}

    virtual void computeError() {
        Sophus::SE3f v1 = (static_cast<VertexPose*>(_vertices[0]))->estimate();
        Eigen::Vector3f v2 = (static_cast<VertexPoint*>(_vertices[1]))->estimate();
        Eigen::Vector3f p = v1.rotationMatrix() * v2 + v1.translation();
        Eigen::Vector2f P;
        p /= p(2);
        P(0) = fx * p(0) + cx;
        P(1) = fy * p(1) + cy;
        _error = _measurement - P;
    }

    virtual void linearizeOplus() {
        Sophus::SE3f v1 = static_cast<VertexPose*>(_vertices[0])->estimate();
        Eigen::Vector3f v2 = (static_cast<VertexPoint*> (_vertices[1]))->estimate();

        Eigen::Vector3f p = v1.so3() * v2 + v1.translation();
        _jacobianOplusXi(0, 0) = fx / p(2);
        _jacobianOplusXi(0, 1) = 0;
        _jacobianOplusXi(0, 2) = -p(0) / p(2) / p(2) * fx;
        _jacobianOplusXi(1, 0) = 0;
        _jacobianOplusXi(1, 1) = fy / p(2);
        _jacobianOplusXi(1, 2) = -p(1) / p(2) / p(2) * fy;

        Eigen::Matrix<float, 3, 6> p_xi;
        p_xi.block(0, 0, 3, 3) = Eigen<float, 3, 3>::Identity();
        p_xi.block(0, 3, 3, 3) = -Sophus::SO3::hat(p);
        _jacobianOplusXj = -_jacobianOplusXi * p_xi;
        _jacobianOplusXi *= -v1.rotationMatrix();
    }

private:
    float fx;
    float fy;
    float cx;
    float cy;
};


int main()
{
    int MaxIter = 50;
    float fx, fy, cx, cy;
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3>> Block;
    Block::LinearSolverType *linearSolver = new g2o::LinearSolverCholmod<Block::PoseMatrixType>();
    Block *solver_ptr = new Block(linearSolver);
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    for(size_t i = 0; i < vertexes.size(); ++i) {
        VertexPose *v = new VertexPose();
        v->setId(vertexes[i].index);
        v->setSE3(vertexes[i].pose);
        optimizer.addVertex(v);
        if(i == 0)
            v->setFixed(true);
    }

    for(size_t i = 0; i < landMarks.size(); ++i) {
        VertexPoint *v = new VertexPoint();
        v->setId(landMarks[i].index);
        v->setPoint(landMarks[i].point);
        optimizer.addVertex(v);
        if(i == 0)
            v->setFixed(true);
    }

    for(size_t i = 0; i < edges.size(); ++i) {
        Edge * e = new Edge();
        e->setId(i);
        e->setVertex(0, optimizer.vertices()[edges[i].i]);
        e->setVertex(1, optimizer.vertices()[edges[i].j]);
        e->setEdge(fx, fy, cx, cy, edges[i]);
        optimizer.addEdge(e);
    }

    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(MaxIter);
}

