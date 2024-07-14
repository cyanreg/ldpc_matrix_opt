/*
 * This file is part of ldpc_matrix_opt.
 *
 * ldpc_matrix_opt is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * ldpc_matrix_opt is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with ldpc_matrix_opt; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <stdio.h>
#include <libavutil/hwcontext_vulkan.h>

#include "vulkan_utils.h"
#include "vulkan_loader.h"

#include "vulkan_spirv.h"

typedef struct MainContext {
    AVBufferRef *dev_ref;
    FFVulkanContext s;
    FFVkSPIRVCompiler *spv;

    int nb_exec_ctx;

    FFVkQueueFamilyCtx qf;
    FFVkExecPool exec_pool;
} MainContext;

typedef struct ShaderContext {
    FFVulkanPipeline pl;
    FFVkSPIRVShader shd;
} ShaderContext;

static int init_vulkan(MainContext *ctx)
{
    int err;
    AVDictionary *opts;
    av_dict_set_int(&opts, "debug", 0, 0);

    err = av_hwdevice_ctx_create(&ctx->dev_ref, AV_HWDEVICE_TYPE_VULKAN,
                                 "0", opts, 0);
    av_dict_free(&opts);

    if (err < 0) {
        printf("Error initializing device: %s\n", av_err2str(err));
        return err;
    }

    /* Initialize context */
    ctx->s.device = (AVHWDeviceContext *)ctx->dev_ref->data;
    ctx->s.hwctx = ctx->s.device->hwctx;

    ctx->s.extensions = ff_vk_extensions_to_mask(ctx->s.hwctx->enabled_dev_extensions,
                                                 ctx->s.hwctx->nb_enabled_dev_extensions);

    err = ff_vk_load_functions(ctx->s.device, &ctx->s.vkfn,
                               ctx->s.extensions, 1, 1);
    if (err < 0) {
        printf("Error loading functions: %s\n", av_err2str(err));
        return err;
    }

    err = ff_vk_load_props(&ctx->s);
    if (err < 0) {
        printf("Error loading device props: %s\n", av_err2str(err));
        return err;
    }

    err = ff_vk_qf_init(&ctx->s, &ctx->qf, VK_QUEUE_COMPUTE_BIT);
    if (err < 0) {
        printf("Error finding queue: %s\n", av_err2str(err));
        return err;
    }

    ctx->nb_exec_ctx = 1;
    err = ff_vk_exec_pool_init(&ctx->s, &ctx->qf, &ctx->exec_pool,
                               ctx->nb_exec_ctx,
                               0, VK_QUERY_TYPE_RESULT_STATUS_ONLY_KHR, 0,
                               NULL);
    if (err < 0) {
        printf("Error initializing execution pool: %s\n", av_err2str(err));
        return err;
    }

    ctx->spv = ff_vk_shaderc_init();
    if (!ctx->spv) {
        printf("Error initializing SPIR-V compiler\n");
        return err;
    }

    return 0;
}

typedef struct ECShaderPush {
    VkDeviceAddress *mat;
} ECShaderPush;

static int init_ec_shader(MainContext *ctx, ShaderContext *sc)
{
    int err;
    FFVkSPIRVShader *shd = &sc->shd;

    err = ff_vk_shader_init(&sc->pl, shd, "ec", VK_SHADER_STAGE_COMPUTE_BIT, 0);
    if (err < 0) {
        printf("Error initializing shader: %s\n", av_err2str(err));
        return err;
    }

    ff_vk_shader_set_compute_sizes(shd, 1, 1, 1);

    GLSLC(0, #extension GL_ARB_gpu_shader_int64 : require                     );
    GLSLC(0,                                                                  );
    GLSLC(0, layout(buffer_reference, buffer_reference_align = 8) buffer MatrixBuffer {  );
    GLSLC(1,     uint64_t v[];                                                );
    GLSLC(0, };                                                               );
    GLSLC(0,                                                                  );
    GLSLC(0, layout(push_constant, std430) uniform pushConstants {            );
    GLSLC(1,     MatrixBuffer matrix_base;                                    );
    GLSLC(0, };                                                               );
    GLSLC(0,                                                                  );

    ff_vk_add_push_constant(&sc->pl, 0, sizeof(ECShaderPush), VK_SHADER_STAGE_COMPUTE_BIT);

    const char *shader_data = {
#include "ec.comp.inc"
    };

    GLSLD(shader_data);

    uint8_t *spv_data;
    size_t spv_len;
    void *spv_opaque = NULL;
    err = ctx->spv->compile_shader(ctx->spv, &ctx->s, shd,
                                   &spv_data, &spv_len,
                                   "main", &spv_opaque);
    if (err < 0) {
        printf("Error compiling shader: %s\n", av_err2str(err));
        return err;
    }

    err = ff_vk_shader_create(&ctx->s, shd, spv_data, spv_len, "main");
    if (err < 0) {
        printf("Error creating shader context: %s\n", av_err2str(err));
        return err;
    }

    err = ff_vk_init_compute_pipeline(&ctx->s, &sc->pl, shd);
    if (err < 0) {
        printf("Error creating pipeline: %s\n", av_err2str(err));
        return err;
    }

    err = ff_vk_exec_pipeline_register(&ctx->s, &ctx->exec_pool, &sc->pl);
    if (err < 0) {
        printf("Error creating pipeline: %s\n", av_err2str(err));
        return err;
    }

    if (spv_opaque)
        ctx->spv->free_shader(ctx->spv, &spv_opaque);

    return 0;
}

static int run_ec_shader(MainContext *ctx, ShaderContext *sc)
{
    int err;
    FFVulkanFunctions *vk = &ctx->s.vkfn;

    FFVkExecContext *exec = ff_vk_exec_get(&ctx->exec_pool);
    ff_vk_exec_start(&ctx->s, exec);

    ff_vk_exec_bind_pipeline(&ctx->s, exec, &sc->pl);
    ECShaderPush pd = {
    };

    ff_vk_update_push_exec(&ctx->s, exec, &sc->pl, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pd), &pd);

    vk->CmdDispatch(exec->buf, 1, 1, 1);

    err = ff_vk_exec_submit(&ctx->s, exec);
    if (err < 0) {
        printf("Error submitting shader: %s\n", av_err2str(err));
        return err;
    }

    ff_vk_exec_wait(&ctx->s, exec);

    return 0;
}

int main(void)
{
    int err;
    MainContext ctx = { };
    err = init_vulkan(&ctx);
    if (err < 0)
        return AVERROR(err);

    ShaderContext sc = { };
    err = init_ec_shader(&ctx, &sc);
    if (err < 0)
        return AVERROR(err);

    err = run_ec_shader(&ctx, &sc);
    if (err < 0)
        return AVERROR(err);

    return 0;
}
