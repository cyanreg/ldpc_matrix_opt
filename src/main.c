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
#include <time.h>

#include "utils.h"
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

    int message_bits;
    int parity_bits;
    int rows_at_once;
} MainContext;

typedef struct ShaderContext {
    FFVulkanPipeline pl;
    FFVkSPIRVShader shd;

    AVBufferPool *msg_pool;
    AVBufferPool *matrix_pool;
} ShaderContext;

static int init_vulkan(MainContext *ctx)
{
    int err;
    AVDictionary *opts = NULL;
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
    VkDeviceAddress mat;
    VkDeviceAddress msg;
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

    GLSLC(0, #extension GL_ARB_gpu_shader_int64 : require                                );
    GLSLC(0, #extension GL_EXT_shader_explicit_arithmetic_types : require                );
    GLSLC(0,                                                                             );
    GLSLF(0, #define message_bits %u                                   ,ctx->message_bits);
    GLSLF(0, #define parity_bits %u                                     ,ctx->parity_bits);
    GLSLC(0,                                                                             );
    GLSLC(0, layout(buffer_reference, buffer_reference_align = 1) buffer OctetBuffer {   );
    GLSLC(1,     uint8_t b[];                                                            );
    GLSLC(0, };                                                                          );
    GLSLC(0, layout(buffer_reference, buffer_reference_align = 8) buffer MatrixBuffer {  );
    GLSLC(1,     uint64_t v[];                                                           );
    GLSLC(0, };                                                                          );
    GLSLC(0,                                                                             );
    GLSLC(0, layout(push_constant, std430) uniform pushConstants {                       );
    GLSLC(1,     MatrixBuffer mat_base;                                                  );
    GLSLC(1,     OctetBuffer msg_base;                                                   );
    GLSLC(0, };                                                                          );
    GLSLC(0,                                                                             );

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

    AVBufferRef *msg_ref;
    err = ff_vk_get_pooled_buffer(&ctx->s, &sc->msg_pool, &msg_ref,
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                  NULL,
                                  (ctx->message_bits + ctx->parity_bits)/8,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (err < 0) {
        printf("Error allocating memory: %s\n", av_err2str(err));
        return err;
    }

    AVBufferRef *mat_ref;
    size_t mat_size;
    mat_size = (ctx->message_bits + ctx->parity_bits)*ctx->parity_bits;
    mat_size /= ctx->rows_at_once;
    mat_size *= (ctx->rows_at_once >> 3);
    err = ff_vk_get_pooled_buffer(&ctx->s, &sc->matrix_pool, &mat_ref,
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                  NULL,
                                  mat_size,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (err < 0) {
        printf("Error allocating memory: %s\n", av_err2str(err));
        return err;
    }

    FFVkExecContext *exec = ff_vk_exec_get(&ctx->exec_pool);
    ff_vk_exec_start(&ctx->s, exec);

    ff_vk_exec_bind_pipeline(&ctx->s, exec, &sc->pl);

    err = ff_vk_exec_add_dep_buf(&ctx->s, exec, &mat_ref, 1, 0);
    if (err < 0) {
        printf("Error adding buffer dep: %s\n", av_err2str(err));
        return err;
    }

    err = ff_vk_exec_add_dep_buf(&ctx->s, exec, &msg_ref, 1, 0);
    if (err < 0) {
        printf("Error adding buffer dep: %s\n", av_err2str(err));
        return err;
    }

    FFVkBuffer *mat_vk = (FFVkBuffer *)mat_ref->data;
    FFVkBuffer *msg_vk = (FFVkBuffer *)msg_ref->data;

    ECShaderPush pd = {
        .mat = mat_vk->address,
        .msg = msg_vk->address,
    };

    ff_vk_update_push_exec(&ctx->s, exec, &sc->pl, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pd), &pd);

    VkBufferMemoryBarrier2 buf_bar[8];
    int nb_buf_bar = 0;

    buf_bar[nb_buf_bar++] = (VkBufferMemoryBarrier2) {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = mat_vk->stage,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = mat_vk->access,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = mat_vk->buf,
        .size = mat_vk->size,
        .offset = 0,
    };

    buf_bar[nb_buf_bar++] = (VkBufferMemoryBarrier2) {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = msg_vk->stage,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = msg_vk->access,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT |
                         VK_ACCESS_2_SHADER_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = msg_vk->buf,
        .size = msg_vk->size,
        .offset = 0,
    };

    vk->CmdPipelineBarrier2(exec->buf, &(VkDependencyInfo) {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pBufferMemoryBarriers = buf_bar,
            .bufferMemoryBarrierCount = nb_buf_bar,
    });
    mat_vk->stage = buf_bar[0].dstStageMask;
    mat_vk->access = buf_bar[0].dstAccessMask;
    msg_vk->stage = buf_bar[1].dstStageMask;
    msg_vk->access = buf_bar[1].dstAccessMask;

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
    ctx.message_bits = 224;
    ctx.parity_bits = 64;
    ctx.rows_at_once = 64;

    av_log_set_level(AV_LOG_TRACE);

    err = init_vulkan(&ctx);
    if (err < 0)
        return AVERROR(err);

    ShaderContext sc = { };
    err = init_ec_shader(&ctx, &sc);
    if (err < 0)
        return AVERROR(err);

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);

    err = run_ec_shader(&ctx, &sc);
    if (err < 0)
        return AVERROR(err);

    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);
    printf("Shader done: %f ms\n", diff_timespec(&ts_end, &ts_start)*1000);

    ff_vk_exec_pool_free(&ctx.s, &ctx.exec_pool);
    ctx.spv->uninit(&ctx.spv);

    av_buffer_pool_uninit(&sc.msg_pool);
    av_buffer_pool_uninit(&sc.matrix_pool);
    ff_vk_pipeline_free(&ctx.s, &sc.pl);
    ff_vk_shader_free(&ctx.s, &sc.shd);

    ff_vk_uninit(&ctx.s);
    av_buffer_unref(&ctx.dev_ref);

    return 0;
}
