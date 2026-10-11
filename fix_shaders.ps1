$content = Get-Content 'AccelerationEngine\GpuVulkan\VulkanShaders.cs' -Raw
# Fix extension names
$content = $content -replace '#extension GL_KHR_subgroup_basic : enable', '#extension GL_KHR_shader_subgroup_basic : enable'
$content = $content -replace '#extension GL_KHR_subgroup_arithmetic : enable', '#extension GL_KHR_shader_subgroup_arithmetic : enable'
# Add shuffle extensions to LayerNorm shaders
$content = $content -replace '(#extension GL_KHR_shader_subgroup_arithmetic : enable\n)layout\(local_size_x = 256\) in;', '$1#extension GL_KHR_shader_subgroup_shuffle : enable\n#extension GL_KHR_shader_subgroup_shuffle_relative : enable\nlayout(local_size_x = 256) in;'
Set-Content -Path 'AccelerationEngine\GpuVulkan\VulkanShaders.cs' -Value $content