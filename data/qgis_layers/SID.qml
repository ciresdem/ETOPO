<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis hasScaleBasedVisibilityFlag="0" maxScale="0" styleCategories="AllStyleCategories" minScale="1e+08" version="3.22.11-Białowieża">
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
      <Option value="false" type="bool" name="WMSBackgroundLayer"/>
      <Option value="false" type="bool" name="WMSPublishDataSourceUrl"/>
      <Option value="0" type="int" name="embeddedWidgets/count"/>
      <Option value="Value" type="QString" name="identify/format"/>
    </Option>
  </customproperties>
  <pipe-data-defined-properties>
    <Option type="Map">
      <Option value="" type="QString" name="name"/>
      <Option name="properties"/>
      <Option value="collection" type="QString" name="type"/>
    </Option>
  </pipe-data-defined-properties>
  <pipe>
    <provider>
      <resampling maxOversampling="2" zoomedOutResamplingMethod="nearestNeighbour" enabled="false" zoomedInResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer opacity="1" nodataColor="" alphaBand="-1" band="1" type="paletted">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <colorPalette>
        <paletteEntry color="#45d0a2" alpha="255" label="8" value="1"/>
        <paletteEntry color="#38f62e" alpha="255" label="9" value="2"/>
        <paletteEntry color="#60439e" alpha="255" label="10" value="3"/>
        <paletteEntry color="#4a8ae4" alpha="255" label="11" value="4"/>
        <paletteEntry color="#6485a7" alpha="255" label="12" value="5"/>
        <paletteEntry color="#078c62" alpha="255" label="13" value="6"/>
        <paletteEntry color="#7c0016" alpha="255" label="14" value="7"/>
        <paletteEntry color="#c18760" alpha="255" label="8" value="8"/>
        <paletteEntry color="#bd7304" alpha="255" label="9" value="9"/>
        <paletteEntry color="#a4a59e" alpha="255" label="10" value="10"/>
        <paletteEntry color="#dcdcdc" alpha="255" label="11" value="11"/>
        <paletteEntry color="#fbfff3" alpha="255" label="12" value="12"/>
        <paletteEntry color="#8e336b" alpha="255" label="13" value="13"/>
      </colorPalette>
      <colorramp type="randomcolors" name="[source]">
        <Option/>
      </colorramp>
    </rasterrenderer>
    <brightnesscontrast contrast="0" brightness="0" gamma="1"/>
    <huesaturation colorizeBlue="128" colorizeGreen="128" colorizeRed="255" invertColors="0" grayscaleMode="0" colorizeStrength="100" saturation="0" colorizeOn="0"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
