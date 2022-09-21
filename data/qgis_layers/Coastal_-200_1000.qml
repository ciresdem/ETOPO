<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis maxScale="0" hasScaleBasedVisibilityFlag="0" version="3.22.11-Białowieża" minScale="1e+08" styleCategories="AllStyleCategories">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
    <Private>0</Private>
  </flags>
  <temporal enabled="0" mode="0" fetchMode="0">
    <fixedRange>
      <start></start>
      <end></end>
    </fixedRange>
  </temporal>
  <customproperties>
    <Option type="Map">
      <Option type="bool" value="false" name="WMSBackgroundLayer"/>
      <Option type="bool" value="false" name="WMSPublishDataSourceUrl"/>
      <Option type="int" value="0" name="embeddedWidgets/count"/>
      <Option type="QString" value="Value" name="identify/format"/>
    </Option>
  </customproperties>
  <pipe-data-defined-properties>
    <Option type="Map">
      <Option type="QString" value="" name="name"/>
      <Option name="properties"/>
      <Option type="QString" value="collection" name="type"/>
    </Option>
  </pipe-data-defined-properties>
  <pipe>
    <provider>
      <resampling enabled="false" zoomedInResamplingMethod="nearestNeighbour" maxOversampling="2" zoomedOutResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer type="singlebandpseudocolor" alphaBand="-1" nodataColor="" band="1" classificationMin="-200" classificationMax="500" opacity="1">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <rastershader>
        <colorrampshader minimumValue="-200" clip="0" maximumValue="500" colorRampType="INTERPOLATED" classificationMode="1" labelPrecision="4">
          <colorramp type="gradient" name="[source]">
            <Option type="Map">
              <Option type="QString" value="18,31,40,255" name="color1"/>
              <Option type="QString" value="255,255,255,255" name="color2"/>
              <Option type="QString" value="0" name="discrete"/>
              <Option type="QString" value="gradient" name="rampType"/>
              <Option type="QString" value="0.285571;137,244,255,255:0.285714;17,132,0,255:0.357143;106,219,83,255:0.428571;255,247,83,255:0.5;253,174,97,255:0.642857;215,25,28,255" name="stops"/>
            </Option>
            <prop v="18,31,40,255" k="color1"/>
            <prop v="255,255,255,255" k="color2"/>
            <prop v="0" k="discrete"/>
            <prop v="gradient" k="rampType"/>
            <prop v="0.285571;137,244,255,255:0.285714;17,132,0,255:0.357143;106,219,83,255:0.428571;255,247,83,255:0.5;253,174,97,255:0.642857;215,25,28,255" k="stops"/>
          </colorramp>
          <item alpha="255" color="#121f28" value="-200" label="-200.0000"/>
          <item alpha="255" color="#89f4ff" value="-0.1" label="-0.1000"/>
          <item alpha="255" color="#118400" value="0" label="0.0000"/>
          <item alpha="255" color="#6adb53" value="50" label="50.0000"/>
          <item alpha="255" color="#fff753" value="100" label="100.0000"/>
          <item alpha="255" color="#fdae61" value="150" label="150.0000"/>
          <item alpha="255" color="#d7191c" value="250" label="250.0000"/>
          <item alpha="255" color="#ffffff" value="500" label="500.0000"/>
          <rampLegendSettings useContinuousLegend="1" direction="0" orientation="2" prefix="" maximumLabel="" minimumLabel="" suffix="">
            <numericFormat id="basic">
              <Option type="Map">
                <Option type="QChar" value="" name="decimal_separator"/>
                <Option type="int" value="6" name="decimals"/>
                <Option type="int" value="0" name="rounding_type"/>
                <Option type="bool" value="false" name="show_plus"/>
                <Option type="bool" value="true" name="show_thousand_separator"/>
                <Option type="bool" value="false" name="show_trailing_zeros"/>
                <Option type="QChar" value="" name="thousand_separator"/>
              </Option>
            </numericFormat>
          </rampLegendSettings>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast gamma="1" contrast="0" brightness="0"/>
    <huesaturation colorizeGreen="128" colorizeBlue="128" colorizeRed="255" colorizeStrength="100" invertColors="0" grayscaleMode="0" saturation="0" colorizeOn="0"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
