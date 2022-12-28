#include "mgr.hpp"
#include "sim.hpp"

#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/physics_assets.hpp>

#include <charconv>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>
#endif

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;
using namespace madrona::py;

namespace GPUHideSeek {

struct Manager::Impl {
    Config cfg;
    PhysicsLoader physicsLoader;
    EpisodeManager *episodeMgr;

    static inline Impl * init(const Config &cfg);
};

struct Manager::CPUImpl {
    EpisodeManager *episodeMgr;
};

#ifdef MADRONA_CUDA_SUPPORT
struct Manager::CUDAImpl : Manager::Impl {
    EpisodeManager *episodeMgr;
    MWCudaExecutor mwGPU;
};
#endif

static void loadPhysicsObjects(PhysicsLoader &loader)
{
    DynArray<RigidBodyMetadata> metadatas(0);
    DynArray<AABB> aabbs(0);
    DynArray<CollisionPrimitive> prims(0);

    { // Sphere:
        metadatas.push_back({
            .invInertiaTensor = { 1.f, 1.f, 1.f },
        });

        aabbs.push_back({
            .pMin = { -1, -1, -1 },
            .pMax = { 1, 1, 1 },
        });

        prims.push_back({
            .type = CollisionPrimitive::Type::Sphere,
            .sphere = {
                .radius = 1.f,
            },
        });
    }

    { // Plane:
        metadatas.push_back({
            .invInertiaTensor = { 1.f, 1.f, 1.f },
        });

        aabbs.push_back({
            .pMin = { -FLT_MAX, -FLT_MAX, -FLT_MAX },
            .pMax = { FLT_MAX, FLT_MAX, FLT_MAX },
        });

        prims.push_back({
            .type = CollisionPrimitive::Type::Plane,
            .plane = {},
        });
    }

    { // Cube:
        metadatas.push_back({
            .invInertiaTensor = { 1.f, 1.f, 1.f },
        });

        aabbs.push_back({
            .pMin = { -1, -1, -1 },
            .pMax = { 1, 1, 1 },
        });

        geometry::HalfEdgeMesh halfEdgeMesh;
        halfEdgeMesh.constructCube();

        prims.push_back({
            .type = CollisionPrimitive::Type::Hull,
            .hull = {
                .halfEdgeMesh = halfEdgeMesh
            },
        });
    }

    loader.loadObjects(metadatas.data(), aabbs.data(),
                       prims.data(), metadatas.size());

}

Manager::Impl * Manager::Impl::init(const Config &cfg)
{
    DynArray<imp::ImportedObject> imported_renderer_objs(0);
    auto sphere_obj = imp::ImportedObject::importObject(
        (std::filesystem::path(DATA_DIR) / "sphere.obj").c_str());

    if (!sphere_obj.has_value()) {
        FATAL("Failed to load sphere");
    }

    imported_renderer_objs.emplace_back(std::move(*sphere_obj));

    auto plane_obj = imp::ImportedObject::importObject(
        (std::filesystem::path(DATA_DIR) / "plane.obj").c_str());

    if (!plane_obj.has_value()) {
        FATAL("Failed to load plane");
    }

    imported_renderer_objs.emplace_back(std::move(*plane_obj));

    auto cube_obj = imp::ImportedObject::importObject(
        (std::filesystem::path(DATA_DIR) / "cube.obj").c_str());

    if (!cube_obj.has_value()) {
        FATAL("Failed to load cube");
    }

    imported_renderer_objs.emplace_back(std::move(*cube_obj));


    switch (cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_CUDA_SUPPORT
        EpisodeManager *episode_mgr = 
            (EpisodeManager *)cu::allocGPU(sizeof(EpisodeManager));
        REQ_CUDA(cudaMemset(episode_mgr, 0, sizeof(EpisodeManager)));

        PhysicsLoader phys_loader(PhysicsLoader::StorageType::CUDA, 10);
        loadPhysicsObjects(phys_loader);


        ObjectManager *phys_obj_mgr = &phys_loader.getObjectManager();

        EpisodeManager *episode_mgr = 
            (EpisodeManager *)cu::allocGPU(sizeof(EpisodeManager));
        REQ_CUDA(cudaMemset(episode_mgr, 0, sizeof(EpisodeManager)));

        HeapArray<WorldInit> world_inits(cfg.numWorlds);

        for (int64_t i = 0; i < (int64_t)cfg.numWorlds; i++) {
            world_inits[i] = WorldInit {
                episode_mgr,
            };
        }

        MWCudaExecutor mwgpu_exec({
            .worldInitPtr = world_inits.data(),
            .numWorldInitBytes = sizeof(WorldInit),
            .numWorldDataBytes = sizeof(Sim),
            .worldDataAlignment = alignof(Sim),
            .numWorlds = cfg.numWorlds,
            .numExportedBuffers = 2,
            .gpuID = (uint32_t)cfg.gpuID,
            .renderWidth = cfg.renderWidth,
            .renderHeight = cfg.renderHeight,
        }, {
            "",
            { GPU_HIDESEEK_SRC_LIST },
            { GPU_HIDESEEK_COMPILE_FLAGS },
            cfg.debugCompile ? CompileConfig::OptMode::Debug :
                CompileConfig::OptMode::LTO,
            CompileConfig::Executor::TaskGraph,
        });

        DynArray<imp::SourceObject> renderer_objects(0);

        for (const auto &imported_obj : imported_renderer_objs) {
            renderer_objects.push_back(imp::SourceObject {
                imported_obj.meshes,
            });
        }

        mwgpu_exec.loadObjects(renderer_objects);

        return new CUDAImpl {
            { 
                cfg,
                std::move(phys_loader),
                episode_mgr,
            },
            std::move(mwgpu_exec),
        };
#else
        FATAL("Madrona was not compiled with CUDA support");
#endif
    } break;
    case ExecMode::CPU: {
        return nullptr;
    } break;
    default: __builtin_unreachable();
    }
}

MADRONA_EXPORT Manager::Manager(const Config &cfg)
    : impl_(Impl::init(cfg))
{}

MADRONA_EXPORT Manager::~Manager() {}

MADRONA_EXPORT void Manager::step()
{
    switch (impl_->cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_CUDA_SUPPORT
        static_cast<CUDAImpl>(impl_)->mwGPU.run();
#endif
    } break;
    case ExecMode::CPU: {
    } break;
    }
}

MADRONA_EXPORT Tensor Manager::resetTensor() const
{
    return exportStateTensor(0, Tensor::ElementType::Int32,
                             {impl_->cfg.numWorlds, 1});
}

MADRONA_EXPORT Tensor Manager::moveActionTensor() const
{
    return exportStateTensor(1, Tensor::ElementType::Int32,
                             {impl_->cfg.numWorlds, 1});
}

MADRONA_EXPORT Tensor Manager::depthTensor() const
{
    void *dev_ptr = nullptr;
    Optional<int> gpu_id = Optional<int>::none();

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        dev_ptr = static_cast<CUDAImpl>(impl_)->mwGPU.depthObservations();
        gpu_id = impl_->cfg.gpuID;
#endif
    } else {
        dev_ptr = nullptr;
    }

    return Tensor(dev_ptr, Tensor::ElementType::Float32,
                     {impl_->cfg.numWorlds, impl_->cfg.renderHeight,
                      impl_->cfg.renderWidth, 1}, gpu_id);
}

MADRONA_EXPORT Tensor Manager::rgbTensor() const
{
    void *dev_ptr = nullptr;
    Optional<int> gpu_id = Optional<int>::none();

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        dev_ptr = static_cast<CUDAImpl>(impl_)->mwGPU.rgbObservations();
        gpu_id = impl_->cfg.gpuID;
#endif
    } else {
        dev_ptr = nullptr;
    }

    return Tensor(dev_ptr, Tensor::ElementType::UInt8,
                  {impl_->cfg.numWorlds, impl_->cfg.renderHeight,
                   impl_->cfg.renderWidth, 4}, gpu_id);
}

Tensor Manager::exportStateTensor(int64_t slot,
                                  Tensor::ElementType type,
                                  Span<const int64_t> dimensions) const
{
    void *dev_ptr = nullptr;
    Optional<int> gpu_id = Optional<int>::none();
    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        dev_ptr = static_cast<CUDAImpl>(impl_)->mwGPU.getExported(slot);
        gpu_id = impl_->cfg.gpuID;
#endif
    } else {
        dev_ptr = nullptr;
    }

    return Tensor(dev_ptr, type, dimensions, gpu_id);
}


}
